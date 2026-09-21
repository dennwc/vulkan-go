package vk

import (
	"runtime"
	"unsafe"

	vulkan "github.com/christerso/vulkan-go/vulkan"
)

// Additional compute-related flag bits.
const (
	// ShaderStageCompute is VkShaderStageFlagBits VK_SHADER_STAGE_COMPUTE_BIT.
	ShaderStageCompute uint32 = 0x00000020

	// Pipeline stage bits (VkPipelineStageFlagBits).
	PipelineStageComputeShaderBit uint32 = 0x00000800
	PipelineStageTransferBit      uint32 = 0x00001000
	PipelineStageHostBit          uint32 = 0x00004000

	// Access bits (VkAccessFlagBits).
	AccessShaderReadBit    uint32 = 0x00000020
	AccessShaderWriteBit   uint32 = 0x00000040
	AccessTransferReadBit  uint32 = 0x00000800
	AccessTransferWriteBit uint32 = 0x00001000
	AccessHostReadBit      uint32 = 0x00002000
	AccessHostWriteBit     uint32 = 0x00004000
	AccessMemoryReadBit    uint32 = 0x00008000
	AccessMemoryWriteBit   uint32 = 0x00010000
)

const (
	stComputePipelineCreateInfo uint32 = 29
	stMemoryBarrier             uint32 = 46
)

// SpecConstant describes one specialization constant for a compute pipeline.
type SpecConstant struct {
	ID  uint32
	Val uint32
}

// ComputePipelineConfig describes a compute pipeline.
type ComputePipelineConfig struct {
	Name   string
	Layout PipelineLayout
	Module ShaderModule
	// Spec holds optional specialization constants, passed as raw uint32 words.
	Spec []SpecConstant
}

// CreateComputePipeline creates a compute pipeline, optionally with
// specialization constants.
func (d Device) CreateComputePipeline(cfg ComputePipelineConfig) (Pipeline, error) {
	name := cstr(cfg.Name)
	stage := vulkan.VkPipelineShaderStageCreateInfo{
		SType:  vulkan.VkStructureType(stPipelineShaderStageCreateInfo),
		Stage:  ShaderStageCompute,
		Module: vulkan.VkShaderModule(cfg.Module),
		PName:  unsafe.Pointer(name),
	}
	var spec vulkan.VkSpecializationInfo
	if len(cfg.Spec) > 0 {
		entries := make([]vulkan.VkSpecializationMapEntry, len(cfg.Spec))
		data := make([]uint32, len(cfg.Spec))
		for i, sc := range cfg.Spec {
			entries[i] = vulkan.VkSpecializationMapEntry{
				ConstantID: sc.ID,
				Offset:     uint32(i * 4),
				Size:       4,
			}
			data[i] = sc.Val
		}
		spec = vulkan.VkSpecializationInfo{
			MapEntryCount: uint32(len(entries)),
			PMapEntries:   unsafe.Pointer(&entries[0]),
			DataSize:      uintptr(len(data)) * 4,
			PData:         unsafe.Pointer(&data[0]),
		}
		stage.PSpecializationInfo = unsafe.Pointer(&spec)
		runtime.KeepAlive(entries)
		runtime.KeepAlive(data)
	}
	ci := vulkan.VkComputePipelineCreateInfo{
		SType:  vulkan.VkStructureType(stComputePipelineCreateInfo),
		Stage:  stage,
		Layout: vulkan.VkPipelineLayout(cfg.Layout),
	}
	var pl vulkan.VkPipeline
	res := Result(vulkan.VkCreateComputePipelines(vulkan.VkDevice(d), 0, 1, unsafe.Pointer(&ci), nil, unsafe.Pointer(&pl)))
	runtime.KeepAlive(name)
	runtime.KeepAlive(&ci)
	if err := res.asError("vkCreateComputePipelines"); err != nil {
		return 0, err
	}
	return Pipeline(pl), nil
}

// Dispatch records a compute dispatch with the given workgroup counts.
func (c CommandBuffer) Dispatch(x, y, z uint32) {
	vulkan.VkCmdDispatch(vulkan.VkCommandBuffer(c), x, y, z)
}

// FillBuffer records a buffer fill with a 4-byte value pattern.
func (c CommandBuffer) FillBuffer(buf Buffer, offset, size DeviceSize, data uint32) {
	vulkan.VkCmdFillBuffer(vulkan.VkCommandBuffer(c), vulkan.VkBuffer(buf), vulkan.VkDeviceSize(offset), vulkan.VkDeviceSize(size), data)
}

// MemoryBarrier records a global memory barrier for shader and transfer
// hazards between two pipeline stages.
func (c CommandBuffer) MemoryBarrier(srcStage, dstStage, srcAccess, dstAccess uint32) {
	mb := vulkan.VkMemoryBarrier{
		SType:         vulkan.VkStructureType(stMemoryBarrier),
		SrcAccessMask: srcAccess,
		DstAccessMask: dstAccess,
	}
	vulkan.VkCmdPipelineBarrier(
		vulkan.VkCommandBuffer(c),
		srcStage, dstStage, 0,
		1, unsafe.Pointer(&mb),
		0, nil,
		0, nil,
	)
	runtime.KeepAlive(&mb)
}

// BindPipelineCompute binds a compute pipeline.
func (c CommandBuffer) BindPipelineCompute(p Pipeline) {
	vulkan.VkCmdBindPipeline(vulkan.VkCommandBuffer(c), vulkan.VkPipelineBindPoint(BindPointCompute), vulkan.VkPipeline(p))
}

// BindDescriptorSetCompute binds a single descriptor set for compute at set 0.
func (c CommandBuffer) BindDescriptorSetCompute(layout PipelineLayout, set DescriptorSet) {
	vs := vulkan.VkDescriptorSet(set)
	vulkan.VkCmdBindDescriptorSets(vulkan.VkCommandBuffer(c), vulkan.VkPipelineBindPoint(BindPointCompute), vulkan.VkPipelineLayout(layout), 0, 1, unsafe.Pointer(&vs), 0, nil)
	runtime.KeepAlive(&vs)
}
