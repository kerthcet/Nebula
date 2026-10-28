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

// Package kubernetes implements provider.Provider on EKS clusters. Each cluster is one region:
// placement picks a cluster, and this adapter launches the real pod's own node there (see
// managedNodes), then creates the pod pinned to it. One NodeClaim maps to one real pod and one
// node, which goes with the pod.
//
// Capacity is EC2's answer to the launch, so a shortage fails Provision itself and the cluster
// is blocklisted like an AWS region; past the launch, the pod waits only for its node to boot.
//
// The real pod lives in the placeholder's namespace, named after the claim like every
// provider's instance, which is what lets the host cluster itself be a target. Nebula reports
// no cost for it: OpenCost covers the clusters.
package kubernetes

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"

	nebulav1alpha1 "github.com/InftyAI/Nebula/api/v1alpha1"
	"github.com/InftyAI/Nebula/pkg/provider"
	"github.com/InftyAI/Nebula/pkg/provider/aws"
	"github.com/InftyAI/Nebula/pkg/provider/catalog"
	"github.com/InftyAI/Nebula/pkg/util"
	logf "sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	pollInterval = 5 * time.Second
	spotNotice   = 2 * time.Minute
	// cleanupTimeout bounds deleting a pod Provision could not finish setting up.
	cleanupTimeout = 10 * time.Second
)

var (
	_ provider.Provider           = (*Provider)(nil)
	_ provider.RegionAccelerators = (*Provider)(nil)
)

type cluster struct {
	cfg     ClusterConfig
	clients clusterClients
	// families maps a canonical accelerator to the instance families carrying it in this
	// cluster's cloud (see clusterCatalogs).
	families map[string][]string
	// nodeTypes sizes the cluster's nodes (see nodeInstanceTypes).
	nodeTypes map[string][]string
}

// Provider is the kubernetes implementation of provider.Provider.
type Provider struct {
	clusters map[string]cluster // by region, the token placement and blocks carry
	order    []string           // config order, which is the order ResolveRegions walks
	// offered is every cluster's nodeTypes merged, for MapAccelerator: placement asks the
	// provider, not a cluster, whether it serves an accelerator.
	offered    map[string][]string
	costLabels []string
	now        func() time.Time // a seam for the node sweep
}

// NewFromFile builds the provider from the cluster table at path, connecting to every
// cluster up front so a bad entry fails at startup rather than on the first Provision.
// costLabels are the --cost-labels keys, copied from each Pod onto its real pod.
func NewFromFile(ctx context.Context, path string, costLabels []string) (*Provider, error) {
	cfg, err := LoadConfig(path)
	if err != nil {
		return nil, err
	}
	cat, err := catalog.Load()
	if err != nil {
		return nil, fmt.Errorf("kubernetes: load catalog: %w", err)
	}
	clients := make(map[string]clusterClients, len(cfg.Clusters))
	for _, cl := range cfg.Clusters {
		cc, err := newEKSClients(ctx, cl)
		if err != nil {
			return nil, err
		}
		clients[cl.Name] = cc
	}
	return New(ctx, cfg.Clusters, clients, cat, costLabels), nil
}

// New returns a Provider over already-built clients, one per cluster in clusters, which
// must already be validated (see Config.Validate). Each cluster keeps only the families its
// region offers, read once here: what a region offers changes in months, and a restart
// picks a change up.
func New(ctx context.Context, clusters []ClusterConfig, clients map[string]clusterClients, cat catalog.Lookup,
	costLabels []string) *Provider {
	p := &Provider{
		clusters:   make(map[string]cluster, len(clusters)),
		offered:    map[string][]string{},
		costLabels: costLabels,
		now:        time.Now,
	}
	for _, cl := range clusters {
		families := familiesByAccelerator(cat.Offerings(clusterCatalogs[cl.Provider]))
		if api := clients[cl.Name].ec2; api != nil {
			families = inRegion(ctx, cl, api, families)
		}
		c := cluster{
			cfg: cl, clients: clients[cl.Name], families: families,
			nodeTypes: nodeInstanceTypes(cat.Offerings(clusterCatalogs[cl.Provider]), families),
		}
		c.clients.nodes.sizes = instanceSizes(ctx, cl, c.clients.nodes, c.nodeTypes)
		p.clusters[cl.Region] = c
		p.order = append(p.order, cl.Region)
		for shape, ts := range c.nodeTypes {
			for _, t := range ts {
				if !slices.Contains(p.offered[shape], t) {
					p.offered[shape] = append(p.offered[shape], t)
				}
			}
		}
	}
	for _, fs := range p.offered {
		slices.Sort(fs)
	}
	return p
}

// familiesByAccelerator turns a catalog's instance types into families per GPU model:
// H100 → [p5] on AWS, where each NVIDIA family carries one GPU model, so the family pins it.
func familiesByAccelerator(rows []provider.Offering) map[string][]string {
	out := make(map[string][]string)
	for _, o := range rows {
		family, _, _ := strings.Cut(o.AcceleratorID, ".")
		key := strings.ToUpper(o.AcceleratorType)
		if family != "" && !slices.Contains(out[key], family) {
			out[key] = append(out[key], family)
		}
	}
	for _, fs := range out {
		slices.Sort(fs)
	}
	return out
}

// inRegion keeps the families of each accelerator that cl's region offers, dropping an
// accelerator with none left. A failed lookup keeps them all: a cluster wrongly ruled out is
// worse than one refused once.
func inRegion(
	ctx context.Context, cl ClusterConfig, api describeOfferings, byAccel map[string][]string,
) map[string][]string {
	var all []string
	for _, fs := range byAccel {
		all = append(all, fs...)
	}
	offered, err := offeredFamilies(ctx, api, all)
	if err != nil {
		logf.FromContext(ctx).Info("kubernetes: could not read the region's instance families; keeping all",
			"cluster", cl.Name, "region", cl.Region, "error", err.Error())
		return byAccel
	}
	out := make(map[string][]string, len(byAccel))
	for accel, fs := range byAccel {
		if kept := slices.DeleteFunc(slices.Clone(fs), func(f string) bool { return !offered[f] }); len(kept) > 0 {
			out[accel] = kept
		}
	}
	return out
}

// Name implements provider.Provider.
func (p *Provider) Name() string { return provider.ProviderKubernetes }

// Capabilities implements provider.Provider.
func (p *Provider) Capabilities() provider.Capabilities {
	return provider.Capabilities{
		SupportsStop:         false,
		SupportsSpot:         true,  // the node's own launch, see managedNodes.ensure
		SupportsEgressPolicy: false, // no NetworkPolicy is written yet
		SupportsCPUOnly:      false, // the catalog sizes GPU nodes only
		NativeTags:           true,
		PreemptionNotice:     spotNotice,
		PollInterval:         pollInterval,
	}
}

// Offerings implements provider.Provider. There is no price table: see the package doc.
func (p *Provider) Offerings(context.Context) ([]provider.Offering, error) { return nil, nil }

// MapAccelerator implements provider.Provider: the instance types, across every cluster, that
// carry exactly count of the accelerator, since each pod gets a node of its own size. A cluster
// that lacks them fails its own Provision with ErrUnsupportedAccelerator (see ensureNode), which
// blocks only that cluster.
func (p *Provider) MapAccelerator(canonical string, count int32) ([]string, bool) {
	ts, ok := p.offered[nodeTypeKey(canonical, count)]
	return slices.Clone(ts), ok
}

// ServesAcceleratorIn implements provider.RegionAccelerators: whether the cluster in region
// has the accelerator's families, so placement skips one whose region does not offer them.
func (p *Provider) ServesAcceleratorIn(canonical, region string) bool {
	cl, ok := p.clusters[region]
	if !ok {
		return false
	}
	_, ok = cl.families[strings.ToUpper(canonical)]
	return ok
}

// Provision implements provider.Provider: launch the pod's node in the cluster req.Region names,
// then create the real pod pinned to it. The node is the capacity, so the result is reserved
// and a shortage fails here.
func (p *Provider) Provision(
	ctx context.Context, pod *corev1.Pod, req provider.ProvisionRequest,
) (provider.ProvisionResult, error) {
	if pod == nil || req.ClaimName == "" {
		return provider.ProvisionResult{}, errors.New("kubernetes: nil pod or empty ClaimName")
	}
	if mode := req.Egress.ModeOrOpen(); mode != nebulav1alpha1.EgressOpen {
		return provider.ProvisionResult{}, fmt.Errorf("kubernetes: cannot enforce egress mode %q yet", mode)
	}
	if req.RegistryAuth != nil {
		// ECR pulls go through the node role; other registries need a pull Secret, not built yet.
		return provider.ProvisionResult{}, req.RegistryAuth.Unsupported(provider.ProviderKubernetes)
	}
	cl, ok := p.clusters[req.Region]
	if !ok {
		return provider.ProvisionResult{}, fmt.Errorf("kubernetes: no cluster in region %q", req.Region)
	}
	nodeName, err := ensureNode(ctx, cl, pod, req)
	if err != nil {
		return provider.ProvisionResult{}, err
	}
	real := buildPod(pod, podSpec{
		claim:      req.ClaimName,
		selection:  cl.clients.nodes.selection(nodeName),
		env:        req.Env.Config,
		hasSecrets: len(req.Env.Secrets) > 0,
		costLabels: p.costLabels,
	})
	kube := cl.clients.kube

	if err := ensureNamespace(ctx, kube, real.Namespace); err != nil {
		cleanup(ctx, cl, "", "", req.ClaimName)
		return provider.ProvisionResult{}, err
	}
	created, adopted, err := createOrAdopt(ctx, cl.clients, real)
	if err != nil {
		cleanup(ctx, cl, "", "", req.ClaimName) // no pod of ours, but a launched node
		return provider.ProvisionResult{}, err
	}
	if len(req.Env.Secrets) > 0 {
		if err := ensureEnvSecret(ctx, kube, created, adopted, req.Env.Secrets); err != nil {
			cleanup(ctx, cl, real.Namespace, real.Name, req.ClaimName)
			return provider.ProvisionResult{}, err
		}
	}
	return provider.ProvisionResult{
		InstanceID: instanceID(req.Region, real.Namespace, real.Name),
		Reserved:   true,
	}, nil
}

// instanceSizes reads the CPU and memory of every type in nodeTypes, logging and returning nil
// on failure so a lost read never rules a type out.
func instanceSizes(
	ctx context.Context, cl ClusterConfig, nodes *managedNodes, nodeTypes map[string][]string,
) map[string]aws.InstanceResources {
	var all []string
	for _, ts := range nodeTypes {
		all = append(all, ts...)
	}
	if len(all) == 0 {
		return nil
	}
	sizes, err := nodes.ec2.InstanceTypes(ctx, all)
	if err != nil {
		logf.FromContext(ctx).Info("kubernetes: could not read instance type sizes; not filtering by CPU or memory",
			"cluster", cl.Name, "error", err.Error())
		return nil
	}
	return sizes
}

// ensureNode launches, or adopts, the managed node for req's pod, sized to it exactly.
func ensureNode(ctx context.Context, cl cluster, pod *corev1.Pod, req provider.ProvisionRequest) (string, error) {
	canonical, count, err := util.AcceleratorRequest(pod)
	if err != nil {
		return "", fmt.Errorf("kubernetes: %w", err)
	}
	if canonical == "" || count <= 0 {
		// The catalog sizes GPU instances only, so this cluster cannot serve CPU-only pods.
		return "", fmt.Errorf("kubernetes: managed nodes serve GPU pods only: %w", provider.ErrUnsupportedAccelerator)
	}
	instanceTypes := cl.nodeTypes[nodeTypeKey(canonical, count)]
	if len(instanceTypes) == 0 {
		return "", fmt.Errorf("kubernetes: no instance type with %d %s: %w",
			count, canonical, provider.ErrUnsupportedAccelerator)
	}
	requests, _ := util.PodResources(pod)
	if instanceTypes = cl.clients.nodes.fitting(instanceTypes, requests); len(instanceTypes) == 0 {
		// The pod's own request, not the candidate's capacity: unlabelled, so nothing is blocked.
		return "", fmt.Errorf("kubernetes: no %d×%s instance type holds %.1f vCPU and %d MiB",
			count, canonical, requests.CPU, requests.MemoryMiB)
	}
	return cl.clients.nodes.ensure(ctx, nodeRequest{
		claim:         req.ClaimName,
		namespace:     pod.Namespace,
		instanceTypes: instanceTypes,
		spot:          req.CapacityType == nebulav1alpha1.CapacitySpot,
		diskGiB:       util.PodEphemeralStorageGiB(pod),
	})
}

// cleanup deletes what Provision could not finish setting up, the pod (when name is set) and a
// managed cluster's node for claim, on a context that outlives the caller's.
// A failure is only logged: the Pod fails regardless, and the NodeClaim teardown finds the
// leftover through FindByClaim, or the node sweep does.
func cleanup(ctx context.Context, cl cluster, namespace, name, claim string) {
	ctx, cancel := context.WithTimeout(context.WithoutCancel(ctx), cleanupTimeout)
	defer cancel()
	if name != "" {
		if err := deleteInstance(ctx, cl.clients, namespace, name); err != nil {
			logf.FromContext(ctx).Error(err, "kubernetes: could not delete a half-set-up pod; teardown will retry",
				"namespace", namespace, "pod", name)
		}
	}
	if err := cl.clients.nodes.release(ctx, cl.clients.kube, claim); err != nil {
		logf.FromContext(ctx).Error(err, "kubernetes: could not release a half-set-up node; the sweep will retry",
			"claim", claim)
	}
}

// ensureNamespace creates the Pod's namespace in a remote cluster on first use. In the host
// cluster it already exists.
func ensureNamespace(ctx context.Context, kube kubeClient, name string) error {
	_, err := kube.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
	if err == nil {
		return nil
	}
	if !apierrors.IsNotFound(err) {
		return fmt.Errorf("kubernetes: get namespace %q: %w", name, classifyAPI(err))
	}
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{
		Name:   name,
		Labels: map[string]string{nebulav1alpha1.ManagedByLabel: nebulav1alpha1.ManagedByValue},
	}}
	if _, err := kube.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{}); err != nil &&
		!apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("kubernetes: create namespace %q: %w", name, classifyAPI(err))
	}
	return nil
}

// Terminate implements provider.Provider. The region is redundant with the id.
func (p *Provider) Terminate(ctx context.Context, instanceID, _ string) error {
	if instanceID == "" {
		return nil
	}
	cl, ns, name, err := p.locate(instanceID)
	if err != nil {
		return err
	}
	if err := deleteInstance(ctx, cl.clients, ns, name); err != nil {
		return err
	}
	return cl.clients.nodes.release(ctx, cl.clients.kube, name) // the real pod is named after the claim
}

// Get implements provider.Provider.
func (p *Provider) Get(ctx context.Context, instanceID, _ string) (*provider.Instance, error) {
	cl, ns, name, err := p.locate(instanceID)
	if err != nil {
		return nil, err
	}
	pod, err := cl.clients.kube.CoreV1().Pods(ns).Get(ctx, name, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("kubernetes: get pod %s: %w", instanceID, classifyAPI(err))
	}
	inst := p.toInstance(cl, pod)
	return &inst, nil
}

// List implements provider.Provider. One unreadable cluster fails the whole call: leaving
// its pods out would make the poll loop read every one of them as terminated.
//
// Note: list is expensive for kubernetes compared to other providers, especially with many pods.
func (p *Provider) List(ctx context.Context) ([]provider.Instance, error) {
	var out []provider.Instance
	for _, region := range p.order {
		cl := p.clusters[region]
		// TODO: this may leads to performance issues if there are many pods
		// in the cluster, informer-based listing should be considered for better performance.
		pods, err := p.listPods(ctx, cl, "")
		if err != nil {
			return nil, err
		}
		for i := range pods {
			out = append(out, p.toInstance(cl, &pods[i]))
		}
		// Housekeeping, not a verdict on any pod: a failure leaves the pods as listed.
		if err := cl.clients.nodes.reconcile(ctx, cl.clients.kube); err != nil {
			logf.FromContext(ctx).Info("kubernetes: could not cordon new nodes; retrying next poll",
				"cluster", cl.cfg.Name, "error", err.Error())
		}
		cl.clients.nodes.sweep(ctx, cl.clients.kube, liveClaims(pods), p.now())
	}
	return out, nil
}

// FindByClaim implements provider.Provider. region names the claim's cluster; an unknown one
// is searched for everywhere rather than reported absent, since absent means "nothing to tear down".
func (p *Provider) FindByClaim(ctx context.Context, claimName, region string) (*provider.Instance, error) {
	regions := p.order
	if _, ok := p.clusters[region]; ok {
		regions = []string{region}
	}
	for _, r := range regions {
		pods, err := p.listPods(ctx, p.clusters[r], fields.OneTermEqualSelector("metadata.name", claimName).String())
		if err != nil {
			return nil, err
		}
		for i := range pods {
			// Checked again here: the selector is the server's job, and not every client honours it.
			if pods[i].Name == claimName {
				inst := p.toInstance(p.clusters[r], &pods[i])
				return &inst, nil
			}
		}
		if inst, err := p.findNode(ctx, p.clusters[r], claimName); inst != nil || err != nil {
			return inst, err
		}
	}
	return nil, nil
}

// findNode reports a managed node with no real pod, which a Provision interrupted between
// launch and create leaves: absent would tell teardown there is nothing to terminate.
func (p *Provider) findNode(ctx context.Context, cl cluster, claim string) (*provider.Instance, error) {
	node, err := cl.clients.nodes.ec2.Find(ctx, nodeTagKey, claim)
	if err != nil || node == nil {
		return nil, err
	}
	return &provider.Instance{
		ID:        instanceID(cl.cfg.Region, node.Tags[nodeNamespaceTagKey], claim),
		ClaimName: claim,
		State:     provider.InstancePending,
		Region:    cl.cfg.Region,
	}, nil
}

// listPods lists this provider's pods in every namespace of cl, narrowed by fieldSelector.
func (p *Provider) listPods(ctx context.Context, cl cluster, fieldSelector string) ([]corev1.Pod, error) {
	list, err := cl.clients.kube.CoreV1().Pods(metav1.NamespaceAll).List(ctx, metav1.ListOptions{
		LabelSelector: labels.Set(ownerLabels()).String(),
		FieldSelector: fieldSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("kubernetes: list pods in cluster %q: %w", cl.cfg.Name, classifyAPI(err))
	}
	return list.Items, nil
}

func (p *Provider) locate(id string) (cluster, string, string, error) {
	region, ns, name, err := parseInstanceID(id)
	if err != nil {
		return cluster{}, "", "", err
	}
	cl, ok := p.clusters[region]
	if !ok {
		return cluster{}, "", "", fmt.Errorf("kubernetes: instance %q is in region %q, which has no cluster", id, region)
	}
	return cl, ns, name, nil
}

// ClassifyProvisionError implements provider.Provider. Every block is confined to one region,
// that is one cluster, auth failures included.
func (p *Provider) ClassifyProvisionError(err error, accelerator, region string) provider.BlockScope {
	if err == nil {
		return provider.BlockScope{}
	}
	scope := provider.ClassifyError(err, provider.TierOf(err), accelerator)
	if scope == (provider.BlockScope{}) {
		return scope // block nothing; decorating it would create a block
	}
	if scope.Deny == provider.DenyAll {
		// A 401/403 is one cluster's access entry or RBAC, not shared by the others.
		return provider.BlockScope{Deny: provider.DenyRegion, Region: region}
	}
	scope.Region = region
	return scope
}

// ResolveRegions implements provider.Provider. Candidates are the clusters' AWS regions, one
// cluster each (see Config.Validate), so a shortage blocks that region, as EC2 capacity is the
// region's rather than a cluster's. A declared value matches a cluster's Geography or its exact
// Region, both from the cluster table; regions come back in declared order, then config order.
//
//	nil/[]        => every cluster
//	["us"]        => clusters with geography us
//	["us-east-1"] => clusters in us-east-1
//
// narrowTo, the Pod's geographies, keeps only clusters with one of them.
func (p *Provider) ResolveRegions(declared, narrowTo []string) []string {
	var within []string
	for _, g := range narrowTo {
		if g = strings.ToLower(strings.TrimSpace(g)); provider.IsGeography(g) {
			within = append(within, g)
		}
	}
	tokens := declared
	if len(tokens) == 0 {
		tokens = []string{""} // matches every cluster
	}
	var out []string
	for _, tok := range tokens {
		tok = strings.TrimSpace(tok)
		for _, region := range p.order {
			cl := p.clusters[region].cfg
			if matches(cl, tok) && (len(within) == 0 || slices.Contains(within, cl.Geography)) &&
				!slices.Contains(out, region) {
				out = append(out, region)
			}
		}
	}
	return out
}

func matches(cl ClusterConfig, token string) bool {
	return token == "" || token == cl.Region || strings.EqualFold(token, cl.Geography)
}

// instanceID is "<region>/<namespace>/<pod>": the region names the cluster, and is part of the
// id because Get and Terminate are handed nothing else that names it.
func instanceID(region, namespace, name string) string {
	return region + "/" + namespace + "/" + name
}

func parseInstanceID(id string) (region, namespace, name string, err error) {
	parts := strings.SplitN(id, "/", 3)
	if len(parts) != 3 || parts[0] == "" || parts[1] == "" || parts[2] == "" {
		return "", "", "", fmt.Errorf("kubernetes: malformed instance id %q", id)
	}
	return parts[0], parts[1], parts[2], nil
}

// createOrAdopt creates the real pod, or adopts the live one an interrupted Provision left
// (adopted=true). Anything else holding the pod's or env Secret's name is refused, not
// replaced: a new pod could start on a warm node with whatever Secret of that name exists.
func createOrAdopt(
	ctx context.Context, cc clusterClients, want *corev1.Pod,
) (pod *corev1.Pod, adopted bool, err error) {
	pods := cc.kube.CoreV1().Pods(want.Namespace)
	got, err := pods.Get(ctx, want.Name, metav1.GetOptions{})
	switch {
	case apierrors.IsNotFound(err):
	case err != nil:
		return nil, false, fmt.Errorf("kubernetes: get pod %s/%s: %w", want.Namespace, want.Name, classifyAPI(err))
	case !ownedByUs(got):
		return nil, false, fmt.Errorf("kubernetes: pod %s/%s exists and is not Nebula's", want.Namespace, want.Name)
	case got.DeletionTimestamp == nil && !util.Finished(got.Status.Phase):
		return got, true, nil
	default:
		return nil, false, fmt.Errorf("kubernetes: pod %s/%s from an earlier sandbox is still present",
			want.Namespace, want.Name)
	}

	secret := envSecretName(want.Name)
	_, err = cc.kube.CoreV1().Secrets(want.Namespace).Get(ctx, secret, metav1.GetOptions{})
	switch {
	case err == nil:
		return nil, false, fmt.Errorf("kubernetes: secret %s/%s already exists", want.Namespace, secret)
	case !apierrors.IsNotFound(err):
		return nil, false, fmt.Errorf("kubernetes: get secret %s/%s: %w", want.Namespace, secret, classifyAPI(err))
	}

	created, err := pods.Create(ctx, want, metav1.CreateOptions{})
	if err != nil {
		return nil, false, fmt.Errorf("kubernetes: create pod %s/%s: %w", want.Namespace, want.Name, classifyAPI(err))
	}
	return created, false, nil
}

// ensureEnvSecret writes the Secret-sourced env, owned by the pod so Kubernetes deletes it with
// it. An existing one is accepted only for an adopted pod, and only if it is Nebula's.
func ensureEnvSecret(ctx context.Context, kube kubeClient, pod *corev1.Pod, adopted bool, env map[string]string) error {
	want := envSecret(pod.Namespace, pod.Name, env)
	want.OwnerReferences = []metav1.OwnerReference{{
		APIVersion: "v1", Kind: "Pod", Name: pod.Name, UID: pod.UID,
	}}
	secrets := kube.CoreV1().Secrets(pod.Namespace)
	_, err := secrets.Create(ctx, want, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		var got *corev1.Secret
		switch got, err = secrets.Get(ctx, want.Name, metav1.GetOptions{}); {
		case err != nil:
		case !adopted:
			err = fmt.Errorf("secret %s/%s appeared before this pod's", want.Namespace, want.Name)
		case !ownedByUs(got):
			err = fmt.Errorf("secret %s/%s exists and is not Nebula's", want.Namespace, want.Name)
		}
	}
	if err != nil {
		return fmt.Errorf("kubernetes: write env secret for %s/%s: %w", pod.Namespace, pod.Name, classifyAPI(err))
	}
	return nil
}

// deleteInstance removes the real pod; already gone is success. Its env Secret follows
// through the ownerReference (see ensureEnvSecret).
func deleteInstance(ctx context.Context, cc clusterClients, namespace, name string) error {
	err := cc.kube.CoreV1().Pods(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return fmt.Errorf("kubernetes: delete pod %s/%s: %w", namespace, name, classifyAPI(err))
	}
	return nil
}

// toInstance normalizes a real pod. Region is the cluster's AWS region, the token the claim
// carries.
func (p *Provider) toInstance(cl cluster, pod *corev1.Pod) provider.Instance {
	return provider.Instance{
		ID:        instanceID(cl.cfg.Region, pod.Namespace, pod.Name),
		ClaimName: pod.Name,
		State:     toState(pod),
		Region:    cl.cfg.Region,
		Endpoint:  pod.Status.PodIP,
	}
}

// toState maps the real pod's phase. Pending covers the whole wait for its node, which exists
// from Provision on: the readiness deadline bounds a boot that never ends. Unknown stays
// Pending, so a node that stopped reporting keeps the poll loop watching rather than going
// terminal on a pod that may still run.
func toState(pod *corev1.Pod) provider.InstanceState {
	switch pod.Status.Phase {
	case corev1.PodRunning:
		// Running only says the containers started; Ready is what makes the workload available.
		// A terminating pod is not available whatever its Ready says: it can read True through
		// the whole grace period while its own cluster already routes nothing to it.
		if podReady(pod) && pod.DeletionTimestamp == nil {
			return provider.InstanceRunning
		}
		return provider.InstanceNotReady
	case corev1.PodSucceeded:
		return provider.InstanceTerminated
	case corev1.PodFailed:
		return provider.InstanceFailed
	default:
		return provider.InstancePending
	}
}

func podReady(pod *corev1.Pod) bool {
	for _, c := range pod.Status.Conditions {
		if c.Type == corev1.PodReady {
			return c.Status == corev1.ConditionTrue
		}
	}
	return false
}

// classifyAPI marks a 401 or 403 as ErrAuth. ClassifyProvisionError narrows that to the one
// cluster, since another cluster's credentials and RBAC are separate.
func classifyAPI(err error) error {
	if apierrors.IsUnauthorized(err) || apierrors.IsForbidden(err) {
		return fmt.Errorf("%w: %w", provider.ErrAuth, err)
	}
	return err
}
