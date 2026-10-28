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
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/yaml"

	"github.com/InftyAI/Nebula/pkg/provider"
	"github.com/InftyAI/Nebula/pkg/provider/aws"
	"github.com/InftyAI/Nebula/pkg/util"
)

// fakeLauncher is EC2 for managed nodes: one live instance per claim.
type fakeLauncher struct {
	live       map[string]*aws.NodeInstance
	launches   []aws.NodeSpec
	launchErr  error
	terminated []string
	sizes      map[string]aws.InstanceResources
}

func (f *fakeLauncher) InstanceTypes(context.Context, []string) (map[string]aws.InstanceResources, error) {
	return f.sizes, nil
}

func (f *fakeLauncher) Launch(_ context.Context, spec aws.NodeSpec) (*aws.NodeInstance, error) {
	f.launches = append(f.launches, spec)
	if f.launchErr != nil {
		return nil, f.launchErr
	}
	claim := spec.Tags[spec.TagKey]
	inst := &aws.NodeInstance{
		ID: "i-" + claim, PrivateDNSName: "ip-" + claim + ".compute.internal",
		Tags: spec.Tags, LaunchTime: time.Now(),
	}
	if f.live == nil {
		f.live = map[string]*aws.NodeInstance{}
	}
	f.live[claim] = inst
	return inst, nil
}

func (f *fakeLauncher) Find(_ context.Context, _, claim string) (*aws.NodeInstance, error) {
	return f.live[claim], nil
}

func (f *fakeLauncher) List(context.Context, string) ([]aws.NodeInstance, error) {
	out := make([]aws.NodeInstance, 0, len(f.live))
	for _, inst := range f.live {
		out = append(out, *inst)
	}
	return out, nil
}

func (f *fakeLauncher) Terminate(_ context.Context, id string) error {
	f.terminated = append(f.terminated, id)
	for claim, inst := range f.live {
		if inst.ID == id {
			delete(f.live, claim)
		}
	}
	return nil
}

var managedCluster = ClusterConfig{
	Name: "eks-use1", Provider: ClusterProviderEKS, Region: "us-east-1", Geography: "us",
	Nodes: &NodesConfig{InstanceProfile: "nebula-node"},
}

// newManagedProvider is one managed cluster over tc's API server and launcher.
func newManagedProvider(tc testCluster, ec2 *fakeLauncher) *Provider {
	cc := tc.clients()
	cc.nodes = &managedNodes{
		ec2:     ec2,
		join:    joinConfig{cluster: "eks-use1", endpoint: "https://api", ca: "Q0E=", serviceCIDR: "172.20.0.0/16"},
		image:   aws.Image{ID: "ami-1"},
		subnets: []aws.Subnet{{ID: "subnet-a", AZ: "us-east-1a"}},
	}
	return New(context.Background(), []ClusterConfig{managedCluster},
		map[string]clusterClients{managedCluster.Name: cc}, testCatalog, nil)
}

func TestManagedProvisionLaunchesItsOwnNode(t *testing.T) {
	tc := newTestCluster(t, false)
	ec2 := &fakeLauncher{}
	p := newManagedProvider(tc, ec2)

	res, err := p.Provision(context.Background(), placeholder(), provisionRequest())
	if err != nil {
		t.Fatal(err)
	}
	// The node is capacity already, so nothing waits on a scheduler to say so.
	if !res.Reserved {
		t.Errorf("result = %+v, want reserved", res)
	}
	if len(ec2.launches) != 1 {
		t.Fatalf("launches = %d, want 1", len(ec2.launches))
	}
	spec := ec2.launches[0]
	// Sized to the pod exactly: 1 H100 is p5.4xlarge, never the 8-GPU p5.48xlarge.
	if !slices.Equal(spec.InstanceTypes, []string{"p5.4xlarge"}) || spec.Spot {
		t.Errorf("launch = %v spot=%v, want [p5.4xlarge] OnDemand", spec.InstanceTypes, spec.Spot)
	}
	if spec.TagKey != nodeTagKey || spec.Tags[nodeTagKey] != "org-1-work-1" || spec.Tags[nodeNamespaceTagKey] != "org-1" {
		t.Errorf("tags = %v", spec.Tags)
	}
	if _, ok := spec.Tags["karpenter.sh/nodepool"]; ok {
		t.Error("a managed node must never carry Karpenter's nodepool tag")
	}

	real, err := tc.kube.CoreV1().Pods("org-1").Get(context.Background(), "org-1-work-1", metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	term := real.Spec.Affinity.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms[0]
	if len(term.MatchFields) != 1 || term.MatchFields[0].Key != "metadata.name" ||
		!slices.Equal(term.MatchFields[0].Values, []string{"ip-org-1-work-1.compute.internal"}) {
		t.Errorf("pin = %+v, want metadata.name of the launched node", term.MatchFields)
	}
	if !slices.ContainsFunc(real.Spec.Tolerations, func(tol corev1.Toleration) bool {
		return tol.Key == corev1.TaintNodeUnschedulable
	}) {
		t.Error("the pod must tolerate its own node's cordon")
	}

	// A retried Provision adopts the node rather than launching another.
	if _, err := p.Provision(context.Background(), placeholder(), provisionRequest()); err != nil {
		t.Fatal(err)
	}
	if len(ec2.launches) != 1 {
		t.Errorf("launches after retry = %d, want 1", len(ec2.launches))
	}
}

// A launch shortage is the launch's own synchronous error, scoped to the candidate.
func TestManagedLaunchShortageBlocksTheCandidate(t *testing.T) {
	tc := newTestCluster(t, false)
	ec2 := &fakeLauncher{launchErr: fmt.Errorf("aws: InsufficientInstanceCapacity: %w", provider.ErrNoCapacity)}
	p := newManagedProvider(tc, ec2)

	_, err := p.Provision(context.Background(), placeholder(), provisionRequest())
	if !errors.Is(err, provider.ErrNoCapacity) {
		t.Fatalf("err = %v, want ErrNoCapacity", err)
	}
	got := p.ClassifyProvisionError(err, "H100:1", "us-east-1")
	if got.Deny != provider.DenyCandidate || got.Region != "us-east-1" {
		t.Errorf("scope = %+v, want the candidate in us-east-1", got)
	}
	_, err = tc.kube.CoreV1().Pods("org-1").Get(context.Background(), "org-1-work-1", metav1.GetOptions{})
	if err == nil {
		t.Error("no pod may be created without its node")
	}
}

func TestManagedNodesServeGPUPodsOnly(t *testing.T) {
	p := newManagedProvider(newTestCluster(t, false), &fakeLauncher{})
	pod := placeholder()
	delete(pod.Labels, "nebula.inftyai.com/accelerator-type")
	pod.Spec.Containers[0].Resources = corev1.ResourceRequirements{}

	_, err := p.Provision(context.Background(), pod, provisionRequest())
	if !errors.Is(err, provider.ErrUnsupportedAccelerator) {
		t.Fatalf("err = %v, want ErrUnsupportedAccelerator", err)
	}
}

func TestManagedTerminateReleasesTheNode(t *testing.T) {
	tc := newTestCluster(t, false)
	ec2 := &fakeLauncher{}
	p := newManagedProvider(tc, ec2)
	res, err := p.Provision(context.Background(), placeholder(), provisionRequest())
	if err != nil {
		t.Fatal(err)
	}
	node := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "ip-org-1-work-1.compute.internal"}}
	if _, err := tc.kube.CoreV1().Nodes().Create(context.Background(), node, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	if err := p.Terminate(context.Background(), res.InstanceID, ""); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(ec2.terminated, []string{"i-org-1-work-1"}) {
		t.Errorf("terminated = %v", ec2.terminated)
	}
	if _, err := tc.kube.CoreV1().Nodes().Get(context.Background(), node.Name, metav1.GetOptions{}); err == nil {
		t.Error("the Node object must go with its instance")
	}
	if err := p.Terminate(context.Background(), res.InstanceID, ""); err != nil {
		t.Errorf("Terminate is not idempotent: %v", err)
	}
}

// A node whose pod was never created must still reach teardown, or nothing terminates it.
func TestManagedFindByClaimSeesANodeWithoutPod(t *testing.T) {
	ec2 := &fakeLauncher{live: map[string]*aws.NodeInstance{
		"org-1-work-1": {ID: "i-1", Tags: map[string]string{nodeNamespaceTagKey: "org-1"}},
	}}
	p := newManagedProvider(newTestCluster(t, false), ec2)

	inst, err := p.FindByClaim(context.Background(), "org-1-work-1", "us-east-1")
	if err != nil || inst == nil || inst.ID != "us-east-1/org-1/org-1-work-1" {
		t.Fatalf("FindByClaim = %+v, %v", inst, err)
	}
}

// An unbound pod on a managed node is booting: it never turns into a capacity block.
func TestManagedUnboundPodIsBootingNotRefused(t *testing.T) {
	tc := newTestCluster(t, false)
	p := newManagedProvider(tc, &fakeLauncher{})
	if _, err := p.Provision(context.Background(), placeholder(), provisionRequest()); err != nil {
		t.Fatal(err)
	}
	p.now = func() time.Time { return time.Now().Add(time.Hour) }
	if got := listState(t, p); got != provider.InstancePending {
		t.Errorf("state = %s, want Pending", got)
	}
}

func TestReconcileCordonsNewNodes(t *testing.T) {
	tc := newTestCluster(t, false)
	p := newManagedProvider(tc, &fakeLauncher{})
	fresh := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "fresh", Labels: map[string]string{managedNodeLabel: labelTrue}},
		Spec: corev1.NodeSpec{Taints: []corev1.Taint{
			{Key: startupTaintKey, Effect: corev1.TaintEffectNoSchedule},
			{Key: "other", Effect: corev1.TaintEffectNoSchedule},
		}},
	}
	theirs := &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: "theirs"},
		Spec: corev1.NodeSpec{Taints: []corev1.Taint{{Key: startupTaintKey, Effect: corev1.TaintEffectNoSchedule}}}}
	for _, n := range []*corev1.Node{fresh, theirs} {
		if _, err := tc.kube.CoreV1().Nodes().Create(context.Background(), n, metav1.CreateOptions{}); err != nil {
			t.Fatal(err)
		}
	}

	if _, err := p.List(context.Background()); err != nil {
		t.Fatal(err)
	}
	got, _ := tc.kube.CoreV1().Nodes().Get(context.Background(), "fresh", metav1.GetOptions{})
	if !got.Spec.Unschedulable || hasStartupTaint(got) || len(got.Spec.Taints) != 1 {
		t.Errorf("fresh node = unschedulable %v, taints %v; want cordoned with only its own taint lifted",
			got.Spec.Unschedulable, got.Spec.Taints)
	}
	other, _ := tc.kube.CoreV1().Nodes().Get(context.Background(), "theirs", metav1.GetOptions{})
	if other.Spec.Unschedulable {
		t.Error("a node that is not Nebula's must be left alone")
	}
}

func TestSweepReleasesOnlyOldOrphans(t *testing.T) {
	now := time.Now()
	ec2 := &fakeLauncher{live: map[string]*aws.NodeInstance{
		"orphan": {ID: "i-orphan", Tags: map[string]string{nodeTagKey: "orphan"}, LaunchTime: now.Add(-time.Hour)},
		"young":  {ID: "i-young", Tags: map[string]string{nodeTagKey: "young"}, LaunchTime: now},
		"live":   {ID: "i-live", Tags: map[string]string{nodeTagKey: "live"}, LaunchTime: now.Add(-time.Hour)},
	}}
	m := &managedNodes{ec2: ec2}
	kube := newTestCluster(t, false).kube

	m.sweep(context.Background(), kube, map[string]bool{"live": true}, now)
	if !slices.Equal(ec2.terminated, []string{"i-orphan"}) {
		t.Fatalf("terminated = %v, want only the old orphan", ec2.terminated)
	}
	// Throttled: a second sweep inside sweepEvery lists nothing.
	ec2.live["orphan2"] = &aws.NodeInstance{
		ID: "i-orphan2", Tags: map[string]string{nodeTagKey: "orphan2"}, LaunchTime: now.Add(-time.Hour),
	}
	m.sweep(context.Background(), kube, nil, now.Add(time.Second))
	if len(ec2.terminated) != 1 {
		t.Errorf("a sweep inside sweepEvery ran: %v", ec2.terminated)
	}
}

func TestNodeUserDataJoinsBehindTheStartupTaint(t *testing.T) {
	m := &managedNodes{join: joinConfig{cluster: "c", endpoint: "https://api", ca: "Q0E=", serviceCIDR: "172.20.0.0/16"}}
	ud, err := m.userData()
	if err != nil {
		t.Fatal(err)
	}
	_, rest, ok := strings.Cut(ud, "Content-Type: application/node.eks.aws\n\n---\n")
	body, _, ok2 := strings.Cut(rest, "\n--NEBULA--")
	if !ok || !ok2 {
		t.Fatalf("not a nodeadm MIME part:\n%s", ud)
	}
	var cfg struct {
		Kind string `json:"kind"`
		Spec struct {
			Cluster map[string]string `json:"cluster"`
			Kubelet struct {
				Config struct {
					RegisterWithTaints []corev1.Taint `json:"registerWithTaints"`
				} `json:"config"`
				Flags []string `json:"flags"`
			} `json:"kubelet"`
		} `json:"spec"`
	}
	if err := yaml.Unmarshal([]byte(body), &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Kind != "NodeConfig" || cfg.Spec.Cluster["name"] != "c" || cfg.Spec.Cluster["cidr"] != "172.20.0.0/16" {
		t.Errorf("config = %+v", cfg)
	}
	taints := cfg.Spec.Kubelet.Config.RegisterWithTaints
	if len(taints) != 1 || taints[0].Key != startupTaintKey {
		t.Errorf("taints = %+v", taints)
	}
	flags := strings.Join(cfg.Spec.Kubelet.Flags, " ")
	for _, want := range []string{managedNodeLabel + "=true", "nvidia.com/gpu.present=true"} {
		if !strings.Contains(flags, want) {
			t.Errorf("flags %q lack %q", flags, want)
		}
	}
}

func TestConfigValidateNodes(t *testing.T) {
	base := ClusterConfig{Name: "a", Provider: "eks", Region: "us-east-1", Geography: "us"}
	noProfile := base
	noProfile.Nodes = &NodesConfig{}
	ok := base
	ok.Nodes = &NodesConfig{InstanceProfile: "p"}

	for name, tc := range map[string]struct {
		cl      ClusterConfig
		wantErr string
	}{
		"with nodes":        {cl: ok},
		"no nodes":          {cl: base, wantErr: "nodes.instanceProfile is required"},
		"no profile in one": {cl: noProfile, wantErr: "nodes.instanceProfile is required"},
	} {
		err := (&Config{Clusters: []ClusterConfig{tc.cl}}).Validate()
		if tc.wantErr == "" && err != nil || tc.wantErr != "" && (err == nil || !strings.Contains(err.Error(), tc.wantErr)) {
			t.Errorf("%s: err = %v, want %q", name, err, tc.wantErr)
		}
	}
}

// The user's own required terms are ORed, so a node pin goes into every one of them.
func TestWithNodeRequirementsKeepsTheNodePin(t *testing.T) {
	user := &corev1.Affinity{NodeAffinity: &corev1.NodeAffinity{
		RequiredDuringSchedulingIgnoredDuringExecution: &corev1.NodeSelector{NodeSelectorTerms: []corev1.NodeSelectorTerm{
			{MatchExpressions: []corev1.NodeSelectorRequirement{{Key: "a", Operator: corev1.NodeSelectorOpExists}}},
			{MatchExpressions: []corev1.NodeSelectorRequirement{{Key: "b", Operator: corev1.NodeSelectorOpExists}}},
		}},
	}}
	got := withNodeRequirements(user, (&managedNodes{}).selection("n1").affinity)
	for i, term := range got.NodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms {
		if len(term.MatchFields) != 1 || term.MatchFields[0].Values[0] != "n1" {
			t.Errorf("term %d lost the pin: %+v", i, term)
		}
	}
}

// A type that cannot hold the pod's CPU or memory is never launched; one of unknown size is.
func TestFittingFiltersBySize(t *testing.T) {
	m := &managedNodes{sizes: map[string]aws.InstanceResources{
		"small": {VCPUs: 16, MemoryMiB: 64 * 1024},
		"large": {VCPUs: 192, MemoryMiB: 2048 * 1024},
	}}
	cases := map[string]struct {
		req  util.Resources
		want []string
	}{
		"fits both":            {util.Resources{CPU: 4, MemoryMiB: 16 * 1024}, []string{"small", "large", "unknown"}},
		"too much memory":      {util.Resources{CPU: 4, MemoryMiB: 63 * 1024}, []string{"large", "unknown"}},
		"too much cpu":         {util.Resources{CPU: 16}, []string{"large", "unknown"}},
		"no request fits both": {util.Resources{}, []string{"small", "large", "unknown"}},
	}
	for name, tc := range cases {
		if got := m.fitting([]string{"small", "large", "unknown"}, tc.req); !slices.Equal(got, tc.want) {
			t.Errorf("%s: fitting = %v, want %v", name, got, tc.want)
		}
	}
}

// A pod too big for every candidate is its own request's problem: no launch, and nothing blocked.
func TestManagedPodTooBigBlocksNothing(t *testing.T) {
	tc := newTestCluster(t, false)
	ec2 := &fakeLauncher{sizes: map[string]aws.InstanceResources{"p5.4xlarge": {VCPUs: 16, MemoryMiB: 256 * 1024}}}
	p := newManagedProvider(tc, ec2)
	pod := placeholder()
	pod.Spec.Containers[0].Resources.Requests[corev1.ResourceMemory] = resource.MustParse("1Ti")

	_, err := p.Provision(context.Background(), pod, provisionRequest())
	if err == nil || len(ec2.launches) != 0 {
		t.Fatalf("err = %v, launches = %d; want an error and no launch", err, len(ec2.launches))
	}
	if got := p.ClassifyProvisionError(err, "H100:1", "us-east-1"); got != (provider.BlockScope{}) {
		t.Errorf("scope = %+v, want nothing blocked", got)
	}
}
