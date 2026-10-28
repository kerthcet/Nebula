/*
Copyright 2026 The InftyAI Team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package kubernetes

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/util/retry"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/yaml"

	nebulav1alpha1 "github.com/InftyAI/Nebula/api/v1alpha1"
	"github.com/InftyAI/Nebula/pkg/provider"
	"github.com/InftyAI/Nebula/pkg/provider/aws"
	"github.com/InftyAI/Nebula/pkg/util"
)

const (
	// nodeTagKey holds the claim on a node's instance. Not aws.ClaimTagKey: the EC2 provider
	// lists every instance with that tag as its own.
	nodeTagKey          = "nebula.inftyai.com/node-claim"
	nodeNamespaceTagKey = "nebula.inftyai.com/node-namespace"
	managedNodeLabel    = "nebula.inftyai.com/managed-node"

	// startupTaintKey keeps every Pod off a node from registration until reconcile cordons it.
	// A cordon is what keeps others off after that, since DaemonSets (the device plugin)
	// tolerate a cordon by default but not this taint.
	startupTaintKey = "nebula.inftyai.com/initializing"

	// eksNvidiaAMI is the EKS-optimized AL2023 NVIDIA AMI for a Kubernetes version: driver and
	// container toolkit included, device plugin not (the cluster's DaemonSet provides it).
	eksNvidiaAMI = "amazon-eks-node-al2023-x86_64-nvidia-%s-*"

	// orphanGrace is how old a node with no real pod must be before the sweep terminates it,
	// well past a Provision's own launch-then-create window.
	orphanGrace = 10 * time.Minute
	sweepEvery  = time.Minute
)

// nodeLauncher is the EC2 side of a managed node, so a test can stand in for aws.NodeLauncher.
type nodeLauncher interface {
	Launch(ctx context.Context, spec aws.NodeSpec) (*aws.NodeInstance, error)
	Find(ctx context.Context, tagKey, claim string) (*aws.NodeInstance, error)
	List(ctx context.Context, tagKey string) ([]aws.NodeInstance, error)
	Terminate(ctx context.Context, id string) error
	InstanceTypes(ctx context.Context, instanceTypes []string) (map[string]aws.InstanceResources, error)
}

// nodeSelection is what pins the real pod to its node (see managedNodes.selection).
type nodeSelection struct {
	affinity    *corev1.Affinity
	tolerations []corev1.Toleration
}

// joinConfig is what nodeadm needs to join a node to the cluster, all from DescribeCluster.
type joinConfig struct {
	cluster     string
	endpoint    string
	ca          string // base64, as DescribeCluster returns it and nodeadm takes it
	serviceCIDR string
}

// managedNodes launches and retires one cluster's nodes: one EC2 instance per real pod, joined
// with nodeadm and deleted with the pod, so capacity is EC2's own synchronous answer.
type managedNodes struct {
	ec2              nodeLauncher
	join             joinConfig
	image            aws.Image
	subnets          []aws.Subnet
	securityGroupIDs []string
	instanceProfile  string
	// sizes is each candidate type's CPU and memory, read once at startup (see fitting); nil
	// when that read failed, which filters nothing.
	sizes map[string]aws.InstanceResources

	mu        sync.Mutex
	lastSweep time.Time
}

// nodeRequest is one real pod's node.
type nodeRequest struct {
	claim, namespace string
	instanceTypes    []string
	spot             bool
	diskGiB          int
}

// ensure returns the name of claim's node, launching it unless a live one exists, which is
// what makes a retried Provision adopt rather than launch twice.
func (m *managedNodes) ensure(ctx context.Context, req nodeRequest) (string, error) {
	inst, err := m.ec2.Find(ctx, nodeTagKey, req.claim)
	if err != nil {
		return "", fmt.Errorf("kubernetes: find node for %s: %w", req.claim, err)
	}
	if inst == nil {
		userData, err := m.userData()
		if err != nil {
			return "", err
		}
		inst, err = m.ec2.Launch(ctx, aws.NodeSpec{
			InstanceTypes:    req.instanceTypes,
			Spot:             req.spot,
			Image:            m.image,
			DiskGiB:          req.diskGiB,
			Subnets:          m.subnets,
			SecurityGroupIDs: m.securityGroupIDs,
			InstanceProfile:  m.instanceProfile,
			UserData:         userData,
			TagKey:           nodeTagKey,
			// Never Karpenter's karpenter.sh/nodepool: its garbage collection terminates every
			// instance carrying that tag that has no NodeClaim of its own.
			Tags: map[string]string{
				nodeTagKey:          req.claim,
				nodeNamespaceTagKey: req.namespace,
				"Name":              "nebula-" + req.claim,
				"kubernetes.io/cluster/" + m.join.cluster: "owned",
			},
		})
		if err != nil {
			return "", fmt.Errorf("kubernetes: launch node for %s: %w", req.claim, err)
		}
	}
	if inst.PrivateDNSName == "" {
		return "", fmt.Errorf("kubernetes: node %s for %s has no private DNS name", inst.ID, req.claim)
	}
	return inst.PrivateDNSName, nil
}

// selection pins the real pod to its own node. The scheduler then binds it only once the node
// has registered, been cordoned and advertises the GPUs, so no wait is written here; the
// cordon toleration is what lets it past the cordon that keeps every other Pod out.
//
// The pin is matchFields on purpose: Karpenter ignores a pod pinned that way (v1.14
// provisioner.validateNodeSelectorTerm) rather than launching a node of its own for it.
func (m *managedNodes) selection(nodeName string) nodeSelection {
	return nodeSelection{
		affinity: &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{
				NodeSelectorTerms: []corev1.NodeSelectorTerm{{MatchFields: []corev1.NodeSelectorRequirement{{
					Key: "metadata.name", Operator: corev1.NodeSelectorOpIn, Values: []string{nodeName},
				}}}},
			},
		}},
		tolerations: []corev1.Toleration{{
			Key: corev1.TaintNodeUnschedulable, Operator: corev1.TolerationOpExists, Effect: corev1.TaintEffectNoSchedule,
		}},
	}
}

// release terminates claim's node and deletes its Node object; nothing to release is success.
// The Node is deleted here rather than left to the cloud controller, so a replacement pod for
// the same claim never meets its predecessor's.
func (m *managedNodes) release(ctx context.Context, kube kubeClient, claim string) error {
	inst, err := m.ec2.Find(ctx, nodeTagKey, claim)
	if err != nil || inst == nil {
		return err
	}
	if err := m.ec2.Terminate(ctx, inst.ID); err != nil {
		return fmt.Errorf("kubernetes: terminate node %s for %s: %w", inst.ID, claim, err)
	}
	if inst.PrivateDNSName != "" {
		err := kube.CoreV1().Nodes().Delete(ctx, inst.PrivateDNSName, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("kubernetes: delete node %s: %w", inst.PrivateDNSName, classifyAPI(err))
		}
	}
	return nil
}

// reconcile cordons every newly registered Nebula node and lifts its startup taint, in that
// order, so no moment exists in which another Pod could be scheduled there.
func (m *managedNodes) reconcile(ctx context.Context, kube kubeClient) error {
	nodes, err := kube.CoreV1().Nodes().List(ctx, metav1.ListOptions{
		LabelSelector: labels.Set{managedNodeLabel: labelTrue}.String(),
	})
	if err != nil {
		return fmt.Errorf("kubernetes: list managed nodes: %w", classifyAPI(err))
	}
	var errs []error
	for i := range nodes.Items {
		if !hasStartupTaint(&nodes.Items[i]) {
			continue
		}
		name := nodes.Items[i].Name
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
			node, err := kube.CoreV1().Nodes().Get(ctx, name, metav1.GetOptions{})
			if err != nil {
				return err
			}
			node.Spec.Unschedulable = true
			node.Spec.Taints = slices.DeleteFunc(node.Spec.Taints, func(t corev1.Taint) bool { return t.Key == startupTaintKey })
			_, err = kube.CoreV1().Nodes().Update(ctx, node, metav1.UpdateOptions{})
			return err
		})
		if err != nil && !apierrors.IsNotFound(err) {
			errs = append(errs, fmt.Errorf("kubernetes: cordon node %s: %w", name, classifyAPI(err)))
		}
	}
	return errors.Join(errs...)
}

func hasStartupTaint(node *corev1.Node) bool {
	return slices.ContainsFunc(node.Spec.Taints, func(t corev1.Taint) bool { return t.Key == startupTaintKey })
}

// sweep releases nodes whose real pod is gone, at most once per sweepEvery: a crash between
// launching a node and creating its pod, or a teardown that failed. live is every claim with a
// real pod in the cluster; a node younger than orphanGrace is left to the Provision launching it.
func (m *managedNodes) sweep(ctx context.Context, kube kubeClient, live map[string]bool, now time.Time) {
	m.mu.Lock()
	due := now.Sub(m.lastSweep) >= sweepEvery
	if due {
		m.lastSweep = now
	}
	m.mu.Unlock()
	if !due {
		return
	}
	log := logf.FromContext(ctx).WithName("kubernetes-nodes").WithValues("cluster", m.join.cluster)
	insts, err := m.ec2.List(ctx, nodeTagKey)
	if err != nil {
		log.Info("could not list managed nodes; orphans wait for the next sweep", "error", err.Error())
		return
	}
	for _, inst := range insts {
		claim := inst.Tags[nodeTagKey]
		if claim == "" || live[claim] || now.Sub(inst.LaunchTime) < orphanGrace {
			continue
		}
		if err := m.release(ctx, kube, claim); err != nil {
			log.Info("could not release an orphaned node", "claim", claim, "instance", inst.ID, "error", err.Error())
			continue
		}
		log.Info("released an orphaned node", "claim", claim, "instance", inst.ID)
	}
}

// userData is nodeadm's NodeConfig: join the cluster labelled as Nebula's, behind the startup
// taint. nvidia.com/gpu.present is what the device plugin's affinity selects; nodeadm sets it
// itself only from kubelet 1.35, so it is set here for older clusters too.
func (m *managedNodes) userData() (string, error) {
	cfg := map[string]any{
		"apiVersion": "node.eks.aws/v1alpha1",
		"kind":       "NodeConfig",
		"spec": map[string]any{
			"cluster": map[string]any{
				"name":                 m.join.cluster,
				"apiServerEndpoint":    m.join.endpoint,
				"certificateAuthority": m.join.ca,
				"cidr":                 m.join.serviceCIDR,
			},
			"kubelet": map[string]any{
				"config": map[string]any{
					"registerWithTaints": []map[string]string{
						{"key": startupTaintKey, "effect": string(corev1.TaintEffectNoSchedule)},
					},
				},
				"flags": []string{"--node-labels=" + strings.Join([]string{
					managedNodeLabel + "=" + labelTrue,
					nebulav1alpha1.ManagedByLabel + "=" + nebulav1alpha1.ManagedByValue,
					"nvidia.com/gpu.present=true",
				}, ",")},
			},
		},
	}
	doc, err := yaml.Marshal(cfg)
	if err != nil {
		return "", fmt.Errorf("kubernetes: render node config: %w", err)
	}
	const boundary = "NEBULA"
	return "MIME-Version: 1.0\nContent-Type: multipart/mixed; boundary=\"" + boundary + "\"\n\n" +
		"--" + boundary + "\nContent-Type: application/node.eks.aws\n\n---\n" + string(doc) +
		"\n--" + boundary + "--\n", nil
}

// nodeInstanceTypes maps "<ACCELERATOR>:<count>" to the instance types carrying exactly that
// many of it, among families: unlike Karpenter's pods, a managed node is sized to its one pod.
func nodeInstanceTypes(rows []provider.Offering, families map[string][]string) map[string][]string {
	out := map[string][]string{}
	for _, o := range rows {
		accel := strings.ToUpper(o.AcceleratorType)
		family, _, _ := strings.Cut(o.AcceleratorID, ".")
		key := nodeTypeKey(accel, o.GPUCount)
		if slices.Contains(families[accel], family) && !slices.Contains(out[key], o.AcceleratorID) {
			out[key] = append(out[key], o.AcceleratorID)
		}
	}
	return out
}

// Node overhead an instance type must leave for the pod: the kubelet's own reservations
// (kube-reserved, eviction threshold) and the DaemonSets every node runs. Approximate on
// purpose, since the real figure is per AMI and per cluster; a type close to the line can
// still leave the pod unbound until the readiness deadline.
const (
	reservedFraction  = 0.1
	minReservedVCPUs  = 0.5
	minReservedMemMiB = 1536
)

// fitting keeps the instance types whose CPU and memory hold req after node overhead, in
// order. A type with no known size is kept: fail open, as for the region's offerings.
func (m *managedNodes) fitting(instanceTypes []string, req util.Resources) []string {
	return slices.DeleteFunc(slices.Clone(instanceTypes), func(t string) bool {
		size, known := m.sizes[t]
		if !known {
			return false
		}
		cpu := float64(size.VCPUs)
		mem := float64(size.MemoryMiB)
		return req.CPU > cpu-max(minReservedVCPUs, cpu*reservedFraction) ||
			float64(req.MemoryMiB) > mem-max(minReservedMemMiB, mem*reservedFraction)
	})
}

func nodeTypeKey(accel string, count int32) string {
	return fmt.Sprintf("%s:%d", strings.ToUpper(accel), count)
}

// liveClaims is the claims with a real pod, for sweep.
func liveClaims(pods []corev1.Pod) map[string]bool {
	out := make(map[string]bool, len(pods))
	for _, p := range pods {
		out[p.Name] = true
	}
	return out
}
