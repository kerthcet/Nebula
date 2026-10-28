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

package aws

import (
	"context"
	"encoding/base64"
	"fmt"
	"strings"
	"time"

	awssdk "github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/ec2"
	ec2types "github.com/aws/aws-sdk-go-v2/service/ec2/types"
)

// Image is an AMI and the root device a launch from it must size (see rootVolume).
type Image struct {
	ID         string
	RootDevice string
	RootGiB    int
}

// Subnet is a launch target and the availability zone it is in.
type Subnet struct {
	ID string
	AZ string
}

// NodeSpec is one Kubernetes node to launch into a cluster's own network, rather than the
// default VPC the EC2 provider uses. TagKey must not be ClaimTagKey: the EC2 provider lists
// every instance carrying that tag as its own.
type NodeSpec struct {
	InstanceTypes    []string
	Spot             bool
	Image            Image
	DiskGiB          int
	Subnets          []Subnet
	SecurityGroupIDs []string
	// InstanceProfile is the node role's instance profile, by name or ARN.
	InstanceProfile string
	// UserData is the plain boot payload (nodeadm's NodeConfig); Launch encodes it.
	UserData string
	Tags     map[string]string
	// TagKey is the tag every node carries, which Find, List and the stale-template sweep
	// key on; Tags[TagKey] is this node's claim.
	TagKey string
}

// NodeInstance is a launched node as EC2 reports it.
type NodeInstance struct {
	ID    string
	State string
	// PrivateDNSName is what the node registers as: an EKS node's name is its private DNS
	// name. Assigned at launch, before the instance boots.
	PrivateDNSName string
	Tags           map[string]string
	LaunchTime     time.Time
}

// NodeLauncher launches and finds Kubernetes nodes on EC2, through the same instant-fleet
// path as RunInstance, so capacity errors are classified the same way.
type NodeLauncher struct {
	c *sdkClient
}

// NewNodeLauncher returns a launcher for cfg's region.
func NewNodeLauncher(cfg awssdk.Config) *NodeLauncher {
	return &NodeLauncher{c: &sdkClient{ec2: ec2.NewFromConfig(cfg), region: cfg.Region}}
}

// Launch launches one node and returns it. A launch with no instance is classified like
// RunInstance's, ErrNoCapacity for a shortage.
func (l *NodeLauncher) Launch(ctx context.Context, spec NodeSpec) (*NodeInstance, error) {
	if spec.TagKey == "" || spec.TagKey == ClaimTagKey || spec.Tags[spec.TagKey] == "" {
		return nil, fmt.Errorf("aws: node launch needs its own tag key, set, other than %s", ClaimTagKey)
	}
	subnets := make([]subnet, len(spec.Subnets))
	for i, s := range spec.Subnets {
		subnets[i] = subnet{id: s.ID, az: s.AZ}
	}
	claim := spec.Tags[spec.TagKey]
	id, err := l.c.launch(ctx, launchInput{
		instanceTypes:    spec.InstanceTypes,
		spot:             spec.Spot,
		image:            spec.Image,
		diskGiB:          spec.DiskGiB,
		subnets:          subnets,
		securityGroupIDs: spec.SecurityGroupIDs,
		instanceProfile:  spec.InstanceProfile,
		userData:         base64.StdEncoding.EncodeToString([]byte(spec.UserData)),
		tags:             spec.Tags,
		tagKey:           spec.TagKey,
		templateName:     "nebula-node-" + claim,
	})
	if err != nil {
		return nil, err
	}
	// The fleet answers with the id only; the private DNS name the node will register as
	// comes from the instance itself.
	inst, err := l.describe(ctx, &ec2.DescribeInstancesInput{InstanceIds: []string{id}})
	if err != nil {
		return nil, fmt.Errorf("aws: describe launched node %s: %w", id, err)
	}
	if len(inst) == 0 || inst[0].PrivateDNSName == "" {
		return nil, fmt.Errorf("aws: launched node %s has no private DNS name", id)
	}
	return &inst[0], nil
}

// Find returns the live node tagged tagKey=claim, or nil if there is none.
func (l *NodeLauncher) Find(ctx context.Context, tagKey, claim string) (*NodeInstance, error) {
	out, err := l.describe(ctx, &ec2.DescribeInstancesInput{Filters: []ec2types.Filter{
		{Name: awssdk.String("tag:" + tagKey), Values: []string{claim}},
		stateFilter(heldStates...),
	}})
	if err != nil || len(out) == 0 {
		return nil, err
	}
	return &out[0], nil
}

// List returns every live node carrying tagKey.
func (l *NodeLauncher) List(ctx context.Context, tagKey string) ([]NodeInstance, error) {
	return l.describe(ctx, &ec2.DescribeInstancesInput{Filters: []ec2types.Filter{
		{Name: awssdk.String("tag-key"), Values: []string{tagKey}},
		stateFilter(heldStates...),
	}})
}

// Terminate terminates a node; an already-gone one is success.
func (l *NodeLauncher) Terminate(ctx context.Context, id string) error {
	return l.c.TerminateInstance(ctx, id)
}

// Image resolves the newest Amazon-owned AMI whose name matches pattern.
func (l *NodeLauncher) Image(ctx context.Context, pattern string) (Image, error) {
	out, err := l.c.ec2.DescribeImages(ctx, &ec2.DescribeImagesInput{
		Owners: []string{"amazon"},
		Filters: []ec2types.Filter{
			{Name: awssdk.String("name"), Values: []string{pattern}},
			{Name: awssdk.String("state"), Values: []string{"available"}},
		},
	})
	if err != nil {
		return Image{}, err
	}
	var newest ec2types.Image
	for _, img := range out.Images {
		if img.ImageId != nil && img.CreationDate != nil &&
			(newest.CreationDate == nil || *img.CreationDate > *newest.CreationDate) {
			newest = img
		}
	}
	if newest.ImageId == nil {
		return Image{}, fmt.Errorf("aws: no AMI named %q in %s", pattern, l.c.region)
	}
	device, size := rootDeviceOf(newest)
	return Image{ID: *newest.ImageId, RootDevice: device, RootGiB: size}, nil
}

// InstanceResources is an instance type's size, as DescribeInstanceTypes reports it.
type InstanceResources struct {
	VCPUs     int32
	MemoryMiB int64
}

// InstanceTypes returns the size of each of types, in calls of at most 100 types each.
func (l *NodeLauncher) InstanceTypes(ctx context.Context, types []string) (map[string]InstanceResources, error) {
	out := make(map[string]InstanceResources, len(types))
	for len(types) > 0 {
		batch := types[:min(len(types), 100)]
		types = types[len(batch):]
		in := &ec2.DescribeInstanceTypesInput{InstanceTypes: make([]ec2types.InstanceType, len(batch))}
		for i, t := range batch {
			in.InstanceTypes[i] = ec2types.InstanceType(t)
		}
		page, err := l.c.ec2.DescribeInstanceTypes(ctx, in)
		if err != nil {
			return nil, err
		}
		for _, it := range page.InstanceTypes {
			var r InstanceResources
			if it.VCpuInfo != nil {
				r.VCPUs = awssdk.ToInt32(it.VCpuInfo.DefaultVCpus)
			}
			if it.MemoryInfo != nil {
				r.MemoryMiB = awssdk.ToInt64(it.MemoryInfo.SizeInMiB)
			}
			out[string(it.InstanceType)] = r
		}
	}
	return out, nil
}

// Subnets resolves ids to subnets with their availability zones.
func (l *NodeLauncher) Subnets(ctx context.Context, ids []string) ([]Subnet, error) {
	out, err := l.c.ec2.DescribeSubnets(ctx, &ec2.DescribeSubnetsInput{SubnetIds: ids})
	if err != nil {
		return nil, err
	}
	subnets := make([]Subnet, 0, len(out.Subnets))
	for _, s := range out.Subnets {
		subnets = append(subnets, Subnet{ID: awssdk.ToString(s.SubnetId), AZ: awssdk.ToString(s.AvailabilityZone)})
	}
	return subnets, nil
}

func (l *NodeLauncher) describe(ctx context.Context, in *ec2.DescribeInstancesInput) ([]NodeInstance, error) {
	var out []NodeInstance
	for {
		page, err := l.c.ec2.DescribeInstances(ctx, in)
		if err != nil {
			if isNotFound(err) {
				return nil, nil
			}
			return nil, err
		}
		for _, r := range page.Reservations {
			for _, inst := range r.Instances {
				n := NodeInstance{
					ID:             awssdk.ToString(inst.InstanceId),
					PrivateDNSName: awssdk.ToString(inst.PrivateDnsName),
					LaunchTime:     awssdk.ToTime(inst.LaunchTime),
					Tags:           map[string]string{},
				}
				if inst.State != nil {
					n.State = string(inst.State.Name)
				}
				for _, t := range inst.Tags {
					n.Tags[awssdk.ToString(t.Key)] = awssdk.ToString(t.Value)
				}
				out = append(out, n)
			}
		}
		if page.NextToken == nil {
			return out, nil
		}
		in.NextToken = page.NextToken
	}
}

// instanceProfileSpec names an instance profile by ARN or by name, whichever was given.
func instanceProfileSpec(profile string) *ec2types.LaunchTemplateIamInstanceProfileSpecificationRequest {
	if strings.HasPrefix(profile, "arn:") {
		return &ec2types.LaunchTemplateIamInstanceProfileSpecificationRequest{Arn: awssdk.String(profile)}
	}
	return &ec2types.LaunchTemplateIamInstanceProfileSpecificationRequest{Name: awssdk.String(profile)}
}
