/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2019 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include <rex/assert.h>
#include <rex/literals.h>
#include <rex/math.h>
#include <rex/ui/d3d12/d3d12_api.h>

namespace rex::graphics::d3d12 {

using namespace ::rex::literals;

class D3D12CommandProcessor;

class DeferredCommandList {
 public:
  DeferredCommandList(const D3D12CommandProcessor& command_processor,
                      size_t initial_size_bytes = 1_MiB);
  ~DeferredCommandList();

  void Reset();
  void Execute(ID3D12GraphicsCommandList* command_list, ID3D12GraphicsCommandList1* command_list_1);

  // Threaded recording (Fable 2 fork): a worker thread replays the commands
  // into the real command list while the GPU thread keeps writing, instead of
  // the whole list being replayed on the GPU thread at the end of the
  // submission. The producer makes complete commands visible with Publish();
  // EndThreadedRecording() publishes the rest and waits until all of it has
  // been recorded, after which the command list can be closed and executed.
  void StartRecordingThread();
  void StopRecordingThread();
  bool IsRecordingThreadRunning() const { return recording_thread_.joinable(); }
  // Call right after Reset(), with the command list reset and open.
  // The recording thread resets the allocator and the command list itself
  // before replaying the first command of the session.
  void BeginThreadedRecording(ID3D12GraphicsCommandList* command_list,
                              ID3D12GraphicsCommandList1* command_list_1,
                              ID3D12CommandAllocator* command_allocator);
  bool IsThreadedRecordingActive() const { return threaded_list_ != nullptr; }
  // Handoffs are batched: commands are only made visible to the recording
  // thread once at least publish_batch_elements_ have accumulated since the
  // last handoff (each handoff is a cross-core signal, too costly per draw).
  // 0 publishes on every call. The rest is always published at the end.
  // Whether this submission is being recorded by the recording thread.
  bool IsThreadedRecording() const { return threaded_list_ != nullptr; }
  void SetPublishBatchBytes(size_t bytes) { publish_batch_elements_ = bytes / sizeof(uintmax_t); }
  void Publish() {
    if (threaded_list_ != nullptr &&
        threaded_session_start_ + command_stream_.size() -
                published_position_.load(std::memory_order_relaxed) >=
            publish_batch_elements_) {
      PublishThreaded();
    }
  }
  void EndThreadedRecording();

  D3D12_RECT* ClearDepthStencilViewAllocatedRects(D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_view,
                                                  D3D12_CLEAR_FLAGS clear_flags, FLOAT depth,
                                                  UINT8 stencil, UINT num_rects) {
    auto args = reinterpret_cast<ClearDepthStencilViewHeader*>(
        WriteCommand(Command::kD3DClearDepthStencilView,
                     sizeof(ClearDepthStencilViewHeader) + num_rects * sizeof(D3D12_RECT)));
    args->depth_stencil_view = depth_stencil_view;
    args->clear_flags = clear_flags;
    args->depth = depth;
    args->stencil = stencil;
    args->num_rects = num_rects;
    return num_rects ? reinterpret_cast<D3D12_RECT*>(args + 1) : nullptr;
  }

  void D3DClearDepthStencilView(D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_view,
                                D3D12_CLEAR_FLAGS clear_flags, FLOAT depth, UINT8 stencil,
                                UINT num_rects, const D3D12_RECT* rects) {
    D3D12_RECT* allocated_rects = ClearDepthStencilViewAllocatedRects(
        depth_stencil_view, clear_flags, depth, stencil, num_rects);
    if (num_rects) {
      assert_not_null(allocated_rects);
      std::memcpy(allocated_rects, rects, num_rects * sizeof(D3D12_RECT));
    }
  }

  void D3DClearRenderTargetView(D3D12_CPU_DESCRIPTOR_HANDLE render_target_view,
                                const FLOAT color_rgba[4], UINT num_rects,
                                const D3D12_RECT* rects) {
    auto args = reinterpret_cast<ClearRenderTargetViewHeader*>(
        WriteCommand(Command::kD3DClearRenderTargetView,
                     sizeof(ClearRenderTargetViewHeader) + num_rects * sizeof(D3D12_RECT)));
    args->render_target_view = render_target_view;
    std::memcpy(args->color_rgba, color_rgba, 4 * sizeof(FLOAT));
    args->num_rects = num_rects;
    if (num_rects != 0) {
      std::memcpy(args + 1, rects, num_rects * sizeof(D3D12_RECT));
    }
  }

  void D3DClearUnorderedAccessViewUint(D3D12_GPU_DESCRIPTOR_HANDLE view_gpu_handle_in_current_heap,
                                       D3D12_CPU_DESCRIPTOR_HANDLE view_cpu_handle,
                                       ID3D12Resource* resource, const UINT values[4],
                                       UINT num_rects, const D3D12_RECT* rects) {
    auto args = reinterpret_cast<ClearUnorderedAccessViewHeader*>(
        WriteCommand(Command::kD3DClearUnorderedAccessViewUint,
                     sizeof(ClearUnorderedAccessViewHeader) + num_rects * sizeof(D3D12_RECT)));
    args->view_gpu_handle_in_current_heap = view_gpu_handle_in_current_heap;
    args->view_cpu_handle = view_cpu_handle;
    args->resource = resource;
    std::memcpy(args->values_uint, values, 4 * sizeof(UINT));
    args->num_rects = num_rects;
    if (num_rects != 0) {
      std::memcpy(args + 1, rects, num_rects * sizeof(D3D12_RECT));
    }
  }

  void D3DCopyBufferRegion(ID3D12Resource* dst_buffer, UINT64 dst_offset,
                           ID3D12Resource* src_buffer, UINT64 src_offset, UINT64 num_bytes) {
    auto& args = *reinterpret_cast<D3DCopyBufferRegionArguments*>(
        WriteCommand(Command::kD3DCopyBufferRegion, sizeof(D3DCopyBufferRegionArguments)));
    args.dst_buffer = dst_buffer;
    args.dst_offset = dst_offset;
    args.src_buffer = src_buffer;
    args.src_offset = src_offset;
    args.num_bytes = num_bytes;
  }

  void D3DCopyResource(ID3D12Resource* dst_resource, ID3D12Resource* src_resource) {
    auto& args = *reinterpret_cast<D3DCopyResourceArguments*>(
        WriteCommand(Command::kD3DCopyResource, sizeof(D3DCopyResourceArguments)));
    args.dst_resource = dst_resource;
    args.src_resource = src_resource;
  }

  void CopyTexture(const D3D12_TEXTURE_COPY_LOCATION& dst, const D3D12_TEXTURE_COPY_LOCATION& src) {
    auto& args = *reinterpret_cast<CopyTextureArguments*>(
        WriteCommand(Command::kCopyTexture, sizeof(CopyTextureArguments)));
    std::memcpy(&args.dst, &dst, sizeof(D3D12_TEXTURE_COPY_LOCATION));
    std::memcpy(&args.src, &src, sizeof(D3D12_TEXTURE_COPY_LOCATION));
  }

  void D3DCopyTextureRegion(const D3D12_TEXTURE_COPY_LOCATION* dst, UINT dst_x, UINT dst_y,
                            UINT dst_z, const D3D12_TEXTURE_COPY_LOCATION* src,
                            const D3D12_BOX* src_box) {
    assert_not_null(dst);
    assert_not_null(src);
    auto& args = *reinterpret_cast<D3DCopyTextureRegionArguments*>(
        WriteCommand(Command::kD3DCopyTextureRegion, sizeof(D3DCopyTextureRegionArguments)));
    std::memcpy(&args.dst, dst, sizeof(D3D12_TEXTURE_COPY_LOCATION));
    args.dst_x = dst_x;
    args.dst_y = dst_y;
    args.dst_z = dst_z;
    std::memcpy(&args.src, src, sizeof(D3D12_TEXTURE_COPY_LOCATION));
    if (src_box) {
      args.has_src_box = true;
      args.src_box = *src_box;
    } else {
      args.has_src_box = false;
    }
  }

  void D3DDispatch(UINT thread_group_count_x, UINT thread_group_count_y,
                   UINT thread_group_count_z) {
    auto& args = *reinterpret_cast<D3DDispatchArguments*>(
        WriteCommand(Command::kD3DDispatch, sizeof(D3DDispatchArguments)));
    args.thread_group_count_x = thread_group_count_x;
    args.thread_group_count_y = thread_group_count_y;
    args.thread_group_count_z = thread_group_count_z;
  }

  void D3DDrawIndexedInstanced(UINT index_count_per_instance, UINT instance_count,
                               UINT start_index_location, INT base_vertex_location,
                               UINT start_instance_location) {
    auto& args = *reinterpret_cast<D3DDrawIndexedInstancedArguments*>(
        WriteCommand(Command::kD3DDrawIndexedInstanced, sizeof(D3DDrawIndexedInstancedArguments)));
    args.index_count_per_instance = index_count_per_instance;
    args.instance_count = instance_count;
    args.start_index_location = start_index_location;
    args.base_vertex_location = base_vertex_location;
    args.start_instance_location = start_instance_location;
  }

  void D3DDrawInstanced(UINT vertex_count_per_instance, UINT instance_count,
                        UINT start_vertex_location, UINT start_instance_location) {
    auto& args = *reinterpret_cast<D3DDrawInstancedArguments*>(
        WriteCommand(Command::kD3DDrawInstanced, sizeof(D3DDrawInstancedArguments)));
    args.vertex_count_per_instance = vertex_count_per_instance;
    args.instance_count = instance_count;
    args.start_vertex_location = start_vertex_location;
    args.start_instance_location = start_instance_location;
  }

  void D3DBeginQuery(ID3D12QueryHeap* query_heap, D3D12_QUERY_TYPE type, UINT index) {
    auto& args = *reinterpret_cast<D3DQueryArguments*>(
        WriteCommand(Command::kD3DBeginQuery, sizeof(D3DQueryArguments)));
    args.query_heap = query_heap;
    args.type = type;
    args.index = index;
  }

  void D3DEndQuery(ID3D12QueryHeap* query_heap, D3D12_QUERY_TYPE type, UINT index) {
    auto& args = *reinterpret_cast<D3DQueryArguments*>(
        WriteCommand(Command::kD3DEndQuery, sizeof(D3DQueryArguments)));
    args.query_heap = query_heap;
    args.type = type;
    args.index = index;
  }

  void D3DResolveQueryData(ID3D12QueryHeap* query_heap, D3D12_QUERY_TYPE type, UINT start_index,
                           UINT num_queries, ID3D12Resource* destination_buffer,
                           UINT64 aligned_destination_buffer_offset) {
    auto& args = *reinterpret_cast<D3DResolveQueryDataArguments*>(
        WriteCommand(Command::kD3DResolveQueryData, sizeof(D3DResolveQueryDataArguments)));
    args.query_heap = query_heap;
    args.type = type;
    args.start_index = start_index;
    args.num_queries = num_queries;
    args.destination_buffer = destination_buffer;
    args.aligned_destination_buffer_offset = aligned_destination_buffer_offset;
  }

  void D3DIASetIndexBuffer(const D3D12_INDEX_BUFFER_VIEW* view) {
    auto& args = *reinterpret_cast<D3D12_INDEX_BUFFER_VIEW*>(
        WriteCommand(Command::kD3DIASetIndexBuffer, sizeof(D3D12_INDEX_BUFFER_VIEW)));
    if (view != nullptr) {
      args.BufferLocation = view->BufferLocation;
      args.SizeInBytes = view->SizeInBytes;
      args.Format = view->Format;
    } else {
      args.BufferLocation = D3D12_GPU_VIRTUAL_ADDRESS(0);
      args.SizeInBytes = 0;
      args.Format = DXGI_FORMAT_UNKNOWN;
    }
  }

  void D3DIASetPrimitiveTopology(D3D12_PRIMITIVE_TOPOLOGY primitive_topology) {
    auto& arg = *reinterpret_cast<D3D12_PRIMITIVE_TOPOLOGY*>(
        WriteCommand(Command::kD3DIASetPrimitiveTopology, sizeof(D3D12_PRIMITIVE_TOPOLOGY)));
    arg = primitive_topology;
  }

  void D3DIASetVertexBuffers(UINT start_slot, UINT num_views,
                             const D3D12_VERTEX_BUFFER_VIEW* views) {
    if (num_views == 0) {
      return;
    }
    static_assert(alignof(D3D12_VERTEX_BUFFER_VIEW) <= alignof(uintmax_t));
    const size_t header_size =
        ::rex::align(sizeof(D3DIASetVertexBuffersHeader), alignof(D3D12_VERTEX_BUFFER_VIEW));
    auto args = reinterpret_cast<D3DIASetVertexBuffersHeader*>(
        WriteCommand(Command::kD3DIASetVertexBuffers,
                     header_size + num_views * sizeof(D3D12_VERTEX_BUFFER_VIEW)));
    args->start_slot = start_slot;
    args->num_views = num_views;
    std::memcpy(reinterpret_cast<uint8_t*>(args) + header_size, views,
                sizeof(D3D12_VERTEX_BUFFER_VIEW) * num_views);
  }

  void D3DOMSetBlendFactor(const FLOAT blend_factor[4]) {
    auto args =
        reinterpret_cast<FLOAT*>(WriteCommand(Command::kD3DOMSetBlendFactor, 4 * sizeof(FLOAT)));
    args[0] = blend_factor[0];
    args[1] = blend_factor[1];
    args[2] = blend_factor[2];
    args[3] = blend_factor[3];
  }

  void D3DOMSetRenderTargets(UINT num_render_target_descriptors,
                             const D3D12_CPU_DESCRIPTOR_HANDLE* render_target_descriptors,
                             BOOL rts_single_handle_to_descriptor_range,
                             const D3D12_CPU_DESCRIPTOR_HANDLE* depth_stencil_descriptor) {
    auto& args = *reinterpret_cast<D3DOMSetRenderTargetsArguments*>(
        WriteCommand(Command::kD3DOMSetRenderTargets, sizeof(D3DOMSetRenderTargetsArguments)));
    num_render_target_descriptors =
        std::min(num_render_target_descriptors, UINT(D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT));
    args.num_render_target_descriptors = num_render_target_descriptors;
    args.rts_single_handle_to_descriptor_range = rts_single_handle_to_descriptor_range ? 1 : 0;
    if (num_render_target_descriptors != 0) {
      std::memcpy(args.render_target_descriptors, render_target_descriptors,
                  (rts_single_handle_to_descriptor_range ? 1 : num_render_target_descriptors) *
                      sizeof(D3D12_CPU_DESCRIPTOR_HANDLE));
    }
    args.depth_stencil = (depth_stencil_descriptor != nullptr) ? 1 : 0;
    if (depth_stencil_descriptor != nullptr) {
      args.depth_stencil_descriptor.ptr = depth_stencil_descriptor->ptr;
    }
  }

  void D3DOMSetStencilRef(UINT stencil_ref) {
    auto& arg = *reinterpret_cast<UINT*>(WriteCommand(Command::kD3DOMSetStencilRef, sizeof(UINT)));
    arg = stencil_ref;
  }

  void D3DResourceBarrier(UINT num_barriers, const D3D12_RESOURCE_BARRIER* barriers) {
    if (num_barriers == 0) {
      return;
    }
    static_assert(alignof(D3D12_RESOURCE_BARRIER) <= alignof(uintmax_t));
    const size_t header_size = ::rex::align(sizeof(UINT), alignof(D3D12_RESOURCE_BARRIER));
    uint8_t* args = reinterpret_cast<uint8_t*>(WriteCommand(
        Command::kD3DResourceBarrier, header_size + num_barriers * sizeof(D3D12_RESOURCE_BARRIER)));
    *reinterpret_cast<UINT*>(args) = num_barriers;
    std::memcpy(args + header_size, barriers, num_barriers * sizeof(D3D12_RESOURCE_BARRIER));
  }

  void RSSetScissorRect(const D3D12_RECT& rect) {
    auto& arg = *reinterpret_cast<D3D12_RECT*>(
        WriteCommand(Command::kRSSetScissorRect, sizeof(D3D12_RECT)));
    arg = rect;
  }

  void RSSetViewport(const D3D12_VIEWPORT& viewport) {
    auto& arg = *reinterpret_cast<D3D12_VIEWPORT*>(
        WriteCommand(Command::kRSSetViewport, sizeof(D3D12_VIEWPORT)));
    arg = viewport;
  }

  void D3DSetComputeRoot32BitConstants(UINT root_parameter_index, UINT num_32bit_values_to_set,
                                       const void* src_data, UINT dest_offset_in_32bit_values) {
    if (num_32bit_values_to_set == 0) {
      return;
    }
    auto args = reinterpret_cast<SetRoot32BitConstantsHeader*>(WriteCommand(
        Command::kD3DSetComputeRoot32BitConstants,
        sizeof(SetRoot32BitConstantsHeader) + num_32bit_values_to_set * sizeof(uint32_t)));
    args->root_parameter_index = root_parameter_index;
    args->num_32bit_values_to_set = num_32bit_values_to_set;
    args->dest_offset_in_32bit_values = dest_offset_in_32bit_values;
    std::memcpy(args + 1, src_data, num_32bit_values_to_set * sizeof(uint32_t));
  }

  void D3DSetGraphicsRoot32BitConstants(UINT root_parameter_index, UINT num_32bit_values_to_set,
                                        const void* src_data, UINT dest_offset_in_32bit_values) {
    if (num_32bit_values_to_set == 0) {
      return;
    }
    auto args = reinterpret_cast<SetRoot32BitConstantsHeader*>(WriteCommand(
        Command::kD3DSetGraphicsRoot32BitConstants,
        sizeof(SetRoot32BitConstantsHeader) + num_32bit_values_to_set * sizeof(uint32_t)));
    args->root_parameter_index = root_parameter_index;
    args->num_32bit_values_to_set = num_32bit_values_to_set;
    args->dest_offset_in_32bit_values = dest_offset_in_32bit_values;
    std::memcpy(args + 1, src_data, num_32bit_values_to_set * sizeof(uint32_t));
  }

  void D3DSetComputeRootConstantBufferView(UINT root_parameter_index,
                                           D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(WriteCommand(
        Command::kD3DSetComputeRootConstantBufferView, sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetGraphicsRootConstantBufferView(UINT root_parameter_index,
                                            D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetGraphicsRootConstantBufferView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetComputeRootDescriptorTable(UINT root_parameter_index,
                                        D3D12_GPU_DESCRIPTOR_HANDLE base_descriptor) {
    auto& args = *reinterpret_cast<SetRootDescriptorTableArguments*>(WriteCommand(
        Command::kD3DSetComputeRootDescriptorTable, sizeof(SetRootDescriptorTableArguments)));
    args.root_parameter_index = root_parameter_index;
    args.base_descriptor.ptr = base_descriptor.ptr;
  }

  void D3DSetGraphicsRootDescriptorTable(UINT root_parameter_index,
                                         D3D12_GPU_DESCRIPTOR_HANDLE base_descriptor) {
    auto& args = *reinterpret_cast<SetRootDescriptorTableArguments*>(WriteCommand(
        Command::kD3DSetGraphicsRootDescriptorTable, sizeof(SetRootDescriptorTableArguments)));
    args.root_parameter_index = root_parameter_index;
    args.base_descriptor.ptr = base_descriptor.ptr;
  }

  void D3DSetComputeRootShaderResourceView(UINT root_parameter_index,
                                           D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(WriteCommand(
        Command::kD3DSetComputeRootShaderResourceView, sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetGraphicsRootShaderResourceView(UINT root_parameter_index,
                                            D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetGraphicsRootShaderResourceView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetComputeRootUnorderedAccessView(UINT root_parameter_index,
                                            D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetComputeRootUnorderedAccessView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetGraphicsRootUnorderedAccessView(UINT root_parameter_index,
                                             D3D12_GPU_VIRTUAL_ADDRESS buffer_location) {
    auto& args = *reinterpret_cast<SetRootConstantBufferViewArguments*>(
        WriteCommand(Command::kD3DSetGraphicsRootUnorderedAccessView,
                     sizeof(SetRootConstantBufferViewArguments)));
    args.root_parameter_index = root_parameter_index;
    args.buffer_location = buffer_location;
  }

  void D3DSetComputeRootSignature(ID3D12RootSignature* root_signature) {
    auto& arg = *reinterpret_cast<ID3D12RootSignature**>(
        WriteCommand(Command::kD3DSetComputeRootSignature, sizeof(ID3D12RootSignature*)));
    arg = root_signature;
  }

  void D3DSetGraphicsRootSignature(ID3D12RootSignature* root_signature) {
    auto& arg = *reinterpret_cast<ID3D12RootSignature**>(
        WriteCommand(Command::kD3DSetGraphicsRootSignature, sizeof(ID3D12RootSignature*)));
    arg = root_signature;
  }

  void SetDescriptorHeaps(ID3D12DescriptorHeap* cbv_srv_uav_descriptor_heap,
                          ID3D12DescriptorHeap* sampler_descriptor_heap) {
    auto& args = *reinterpret_cast<SetDescriptorHeapsArguments*>(
        WriteCommand(Command::kSetDescriptorHeaps, sizeof(SetDescriptorHeapsArguments)));
    args.cbv_srv_uav_descriptor_heap = cbv_srv_uav_descriptor_heap;
    args.sampler_descriptor_heap = sampler_descriptor_heap;
  }

  void D3DSetPipelineState(ID3D12PipelineState* pipeline_state) {
    auto& arg = *reinterpret_cast<ID3D12PipelineState**>(
        WriteCommand(Command::kD3DSetPipelineState, sizeof(ID3D12PipelineState*)));
    arg = pipeline_state;
  }

  void SetPipelineStateHandle(void* pipeline_state_handle) {
    auto& arg =
        *reinterpret_cast<void**>(WriteCommand(Command::kSetPipelineStateHandle, sizeof(void*)));
    arg = pipeline_state_handle;
  }

  void D3DSetSamplePositions(UINT num_samples_per_pixel, UINT num_pixels,
                             const D3D12_SAMPLE_POSITION* sample_positions) {
    auto& args = *reinterpret_cast<D3DSetSamplePositionsArguments*>(
        WriteCommand(Command::kD3DSetSamplePositions, sizeof(D3DSetSamplePositionsArguments)));
    args.num_samples_per_pixel = num_samples_per_pixel;
    args.num_pixels = num_pixels;
    std::memcpy(
        args.sample_positions, sample_positions,
        std::min(num_samples_per_pixel * num_pixels, UINT(16)) * sizeof(D3D12_SAMPLE_POSITION));
  }

  void BeginDebugMarker(const char* label_name) {
    size_t label_len = std::strlen(label_name);
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kBeginDebugMarker, sizeof(DebugMarkerHeader) + label_len + 1));
    auto& args = *reinterpret_cast<DebugMarkerHeader*>(args_ptr);
    args.label_length = static_cast<uint32_t>(label_len);
    std::memcpy(args_ptr + sizeof(DebugMarkerHeader), label_name, label_len + 1);
  }

  void EndDebugMarker() { WriteCommand(Command::kEndDebugMarker, 0); }

  void InsertDebugMarker(const char* label_name) {
    size_t label_len = std::strlen(label_name);
    uint8_t* args_ptr = reinterpret_cast<uint8_t*>(
        WriteCommand(Command::kInsertDebugMarker, sizeof(DebugMarkerHeader) + label_len + 1));
    auto& args = *reinterpret_cast<DebugMarkerHeader*>(args_ptr);
    args.label_length = static_cast<uint32_t>(label_len);
    std::memcpy(args_ptr + sizeof(DebugMarkerHeader), label_name, label_len + 1);
  }

 private:
  enum class Command {
    kD3DClearDepthStencilView,
    kD3DClearRenderTargetView,
    kD3DClearUnorderedAccessViewUint,
    kD3DCopyBufferRegion,
    kD3DCopyResource,
    kCopyTexture,
    kD3DCopyTextureRegion,
    kD3DDispatch,
    kD3DDrawIndexedInstanced,
    kD3DDrawInstanced,
    kD3DBeginQuery,
    kD3DEndQuery,
    kD3DResolveQueryData,
    kD3DIASetIndexBuffer,
    kD3DIASetPrimitiveTopology,
    kD3DIASetVertexBuffers,
    kD3DOMSetBlendFactor,
    kD3DOMSetRenderTargets,
    kD3DOMSetStencilRef,
    kD3DResourceBarrier,
    kRSSetScissorRect,
    kRSSetViewport,
    kD3DSetComputeRoot32BitConstants,
    kD3DSetGraphicsRoot32BitConstants,
    kD3DSetComputeRootConstantBufferView,
    kD3DSetGraphicsRootConstantBufferView,
    kD3DSetComputeRootDescriptorTable,
    kD3DSetGraphicsRootDescriptorTable,
    kD3DSetComputeRootShaderResourceView,
    kD3DSetGraphicsRootShaderResourceView,
    kD3DSetComputeRootSignature,
    kD3DSetGraphicsRootSignature,
    kD3DSetComputeRootUnorderedAccessView,
    kD3DSetGraphicsRootUnorderedAccessView,
    kSetDescriptorHeaps,
    kD3DSetPipelineState,
    kSetPipelineStateHandle,
    kD3DSetSamplePositions,
    kBeginDebugMarker,
    kEndDebugMarker,
    kInsertDebugMarker,
  };

  struct CommandHeader {
    Command command;
    uint32_t arguments_size_elements;
  };
  static constexpr size_t kCommandHeaderSizeElements =
      (sizeof(CommandHeader) + sizeof(uintmax_t) - 1) / sizeof(uintmax_t);

  struct ClearDepthStencilViewHeader {
    D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_view;
    D3D12_CLEAR_FLAGS clear_flags;
    FLOAT depth;
    UINT8 stencil;
    UINT num_rects;
  };

  struct ClearRenderTargetViewHeader {
    D3D12_CPU_DESCRIPTOR_HANDLE render_target_view;
    FLOAT color_rgba[4];
    UINT num_rects;
  };

  struct ClearUnorderedAccessViewHeader {
    D3D12_GPU_DESCRIPTOR_HANDLE view_gpu_handle_in_current_heap;
    D3D12_CPU_DESCRIPTOR_HANDLE view_cpu_handle;
    ID3D12Resource* resource;
    union {
      FLOAT values_float[4];
      UINT values_uint[4];
    };
    UINT num_rects;
  };

  struct D3DCopyBufferRegionArguments {
    ID3D12Resource* dst_buffer;
    UINT64 dst_offset;
    ID3D12Resource* src_buffer;
    UINT64 src_offset;
    UINT64 num_bytes;
  };

  struct D3DCopyResourceArguments {
    ID3D12Resource* dst_resource;
    ID3D12Resource* src_resource;
  };

  struct CopyTextureArguments {
    D3D12_TEXTURE_COPY_LOCATION dst;
    D3D12_TEXTURE_COPY_LOCATION src;
  };

  struct D3DCopyTextureRegionArguments {
    D3D12_TEXTURE_COPY_LOCATION dst;
    UINT dst_x;
    UINT dst_y;
    UINT dst_z;
    D3D12_TEXTURE_COPY_LOCATION src;
    D3D12_BOX src_box;
    bool has_src_box;
  };

  struct D3DDispatchArguments {
    UINT thread_group_count_x;
    UINT thread_group_count_y;
    UINT thread_group_count_z;
  };

  struct D3DDrawIndexedInstancedArguments {
    UINT index_count_per_instance;
    UINT instance_count;
    UINT start_index_location;
    INT base_vertex_location;
    UINT start_instance_location;
  };

  struct D3DDrawInstancedArguments {
    UINT vertex_count_per_instance;
    UINT instance_count;
    UINT start_vertex_location;
    UINT start_instance_location;
  };

  struct D3DQueryArguments {
    ID3D12QueryHeap* query_heap;
    D3D12_QUERY_TYPE type;
    UINT index;
  };

  struct D3DResolveQueryDataArguments {
    ID3D12QueryHeap* query_heap;
    D3D12_QUERY_TYPE type;
    UINT start_index;
    UINT num_queries;
    ID3D12Resource* destination_buffer;
    UINT64 aligned_destination_buffer_offset;
  };

  struct D3DIASetVertexBuffersHeader {
    UINT start_slot;
    UINT num_views;
  };

  struct D3DOMSetRenderTargetsArguments {
    uint8_t num_render_target_descriptors;
    bool rts_single_handle_to_descriptor_range;
    bool depth_stencil;
    D3D12_CPU_DESCRIPTOR_HANDLE
    render_target_descriptors[D3D12_SIMULTANEOUS_RENDER_TARGET_COUNT];
    D3D12_CPU_DESCRIPTOR_HANDLE depth_stencil_descriptor;
  };

  struct SetRoot32BitConstantsHeader {
    UINT root_parameter_index;
    UINT num_32bit_values_to_set;
    UINT dest_offset_in_32bit_values;
  };

  struct SetRootConstantBufferViewArguments {
    UINT root_parameter_index;
    D3D12_GPU_VIRTUAL_ADDRESS buffer_location;
  };

  struct SetRootDescriptorTableArguments {
    UINT root_parameter_index;
    D3D12_GPU_DESCRIPTOR_HANDLE base_descriptor;
  };

  struct SetDescriptorHeapsArguments {
    ID3D12DescriptorHeap* cbv_srv_uav_descriptor_heap;
    ID3D12DescriptorHeap* sampler_descriptor_heap;
  };

  struct D3DSetSamplePositionsArguments {
    UINT num_samples_per_pixel;
    UINT num_pixels;
    D3D12_SAMPLE_POSITION sample_positions[16];
  };

  struct DebugMarkerHeader {
    uint32_t label_length;
    // Followed by null-terminated label string.
  };

  void* WriteCommand(Command command, size_t arguments_size_bytes);

  void ExecuteRange(ID3D12GraphicsCommandList* command_list,
                    ID3D12GraphicsCommandList1* command_list_1, const uintmax_t* stream,
                    size_t stream_remaining, ID3D12PipelineState*& current_pipeline_state,
                    bool wait_for_pipelines) const;
  void PublishThreaded();
  void WaitUntilRecorded(size_t target);
  void RecordingThreadMain();

  const D3D12CommandProcessor& command_processor_;

  // uintmax_t to ensure uint64_t and pointer alignment of all structures.
  std::vector<uintmax_t> command_stream_;

  // Threaded recording. Positions are absolute (they never go back), in
  // uintmax_t elements: the stream offset of a command is its position minus
  // threaded_session_start_, so nothing has to be reset while the worker may be
  // polling. The worker reads command_stream_ through published_data_ only up
  // to published_position_; command_stream_ is never reallocated while the
  // worker may be reading it (WriteCommand waits for it to catch up first).
  // The worker resets, records and closes the command list itself - the driver
  // needs the list closed on the thread that recorded it.
  std::thread recording_thread_;
  ID3D12GraphicsCommandList* threaded_list_ = nullptr;
  ID3D12GraphicsCommandList1* threaded_list_1_ = nullptr;
  ID3D12PipelineState* threaded_pipeline_state_ = nullptr;
  ID3D12CommandAllocator* threaded_allocator_ = nullptr;
  bool threaded_list_reset_ = false;  // touched by the recording thread only
  size_t threaded_session_start_ = 0;
  size_t publish_batch_elements_ = 0;
  std::atomic<const uintmax_t*> published_data_{nullptr};
  std::atomic<size_t> published_position_{0};
  std::atomic<size_t> recorded_position_{0};
  std::atomic<bool> recording_thread_sleeping_{false};
  std::atomic<bool> recording_thread_stop_{false};
  std::atomic<bool> close_requested_{false};
  std::atomic<bool> close_done_{false};
  std::mutex recording_mutex_;
  std::condition_variable recording_cond_;
};

}  // namespace rex::graphics::d3d12
