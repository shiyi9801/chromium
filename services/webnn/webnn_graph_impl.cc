// Copyright 2023 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include "services/webnn/webnn_graph_impl.h"

#include <math.h>

#include <algorithm>
#include <utility>
#include <vector>

#include "base/dcheck_is_on.h"
#include "base/types/optional_ref.h"
#include "base/types/pass_key.h"
#include "services/webnn/compute_resource_info.pb.h"
#include "services/webnn/error.h"
#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/webnn_context_impl.h"
#include "services/webnn/webnn_tensor_impl.h"

namespace webnn {

namespace {

// Return false if the named tensors for dispatch don't match the built
// graph's expectation.
bool ValidateWebNNTensors(
    const base::flat_map<std::string_view, WebNNTensorImpl*>& named_tensors,
    const base::flat_map<std::string, OperandDescriptor>&
        names_to_descriptors) {
  return std::ranges::equal(
      named_tensors, names_to_descriptors,
      [](const auto& named_tensor, const auto& tensor_spec) {
        const auto& [tensor_name, tensor_impl] = named_tensor;
        const auto& [tensor_spec_name, tensor_spec_descriptor] = tensor_spec;
        return tensor_name == tensor_spec_name &&
               tensor_impl->data_type() == tensor_spec_descriptor.data_type() &&
               tensor_impl->shape() == tensor_spec_descriptor.shape();
      });
}

// Return false if the same tensor was specified in inputs and outputs.
bool ValidateWebNNTensorsUsage(
    const base::flat_map<std::string, blink::WebNNTensorToken>& named_inputs,
    const base::flat_map<std::string, blink::WebNNTensorToken>& named_outputs) {
  // Validate that output tensors are unique.
  std::set<blink::WebNNTensorToken> output_tensors;
  for (const auto& named_output : named_outputs) {
    output_tensors.insert(named_output.second);
  }

  if (output_tensors.size() != named_outputs.size()) {
    return false;
  }

  // Validate tensors used for input and output are unique.
  for (const auto& named_input : named_inputs) {
    if (output_tensors.contains(named_input.second)) {
      return false;
    }
  }

  return true;
}

}  // namespace

WebNNGraphImpl::ComputeResourceInfo::ComputeResourceInfo(
    base::flat_map<std::string, OperandDescriptor> input_names_to_descriptors,
    base::flat_map<std::string, OperandDescriptor> output_names_to_descriptors,
    base::flat_map<uint64_t, base::flat_set<size_t>>
        operand_to_dependent_operations,
    base::PassKey<WebNNGraphBuilderImpl> pass_key)
    : input_names_to_descriptors(std::move(input_names_to_descriptors)),
      output_names_to_descriptors(std::move(output_names_to_descriptors)),
      operand_to_dependent_operations(
          std::move(operand_to_dependent_operations)) {}

WebNNGraphImpl::ComputeResourceInfo::ComputeResourceInfo(
    base::flat_map<std::string, OperandDescriptor> input_names_to_descriptors,
    base::flat_map<std::string, OperandDescriptor> output_names_to_descriptors,
    base::flat_map<uint64_t, base::flat_set<size_t>>
        operand_to_dependent_operations)
    : input_names_to_descriptors(std::move(input_names_to_descriptors)),
      output_names_to_descriptors(std::move(output_names_to_descriptors)),
      operand_to_dependent_operations(
          std::move(operand_to_dependent_operations)) {}

WebNNGraphImpl::ComputeResourceInfo::ComputeResourceInfo(
    ComputeResourceInfo&&) = default;
WebNNGraphImpl::ComputeResourceInfo&
WebNNGraphImpl::ComputeResourceInfo::operator=(ComputeResourceInfo&&) = default;

WebNNGraphImpl::ComputeResourceInfo::~ComputeResourceInfo() = default;

bool WebNNGraphImpl::ComputeResourceInfo::SerializeToString(
    std::string& str) const {
  services::webnn::proto::ComputeResourceInfo resource_info_proto;

  auto* input_names_to_descriptors_proto =
      resource_info_proto.mutable_input_names_to_descriptors();
  auto* output_names_to_descriptors_proto =
      resource_info_proto.mutable_output_names_to_descriptors();
  auto* operand_to_dependent_operations_proto =
      resource_info_proto.mutable_operand_to_dependent_operations();

  auto convert_data_type = [](OperandDataType data_type) {
    switch (data_type) {
      case OperandDataType::kFloat32:
        return services::webnn::proto::OperandDataType::kFloat32;
      case OperandDataType::kFloat16:
        return services::webnn::proto::OperandDataType::kFloat16;
      case OperandDataType::kInt32:
        return services::webnn::proto::OperandDataType::kInt32;
      case OperandDataType::kUint32:
        return services::webnn::proto::OperandDataType::kUint32;
      case OperandDataType::kInt64:
        return services::webnn::proto::OperandDataType::kInt64;
      case OperandDataType::kUint64:
        return services::webnn::proto::OperandDataType::kUint64;
      case OperandDataType::kInt8:
        return services::webnn::proto::OperandDataType::kInt8;
      case OperandDataType::kUint8:
        return services::webnn::proto::OperandDataType::kUint8;
      case OperandDataType::kInt4:
        return services::webnn::proto::OperandDataType::kInt4;
      case OperandDataType::kUint4:
        return services::webnn::proto::OperandDataType::kUint4;
    }
  };

  for (const auto& [name, descriptor] : input_names_to_descriptors) {
    services::webnn::proto::OperandDescriptor descriptor_proto;
    for (uint32_t dim : descriptor.shape()) {
      descriptor_proto.add_dim(dim);
    }
    descriptor_proto.set_data_type(convert_data_type(descriptor.data_type()));
    input_names_to_descriptors_proto->emplace(name,
                                              std::move(descriptor_proto));
  }

  for (const auto& [name, descriptor] : output_names_to_descriptors) {
    services::webnn::proto::OperandDescriptor descriptor_proto;
    for (uint32_t dim : descriptor.shape()) {
      descriptor_proto.add_dim(dim);
    }
    descriptor_proto.set_data_type(convert_data_type(descriptor.data_type()));
    output_names_to_descriptors_proto->emplace(name,
                                               std::move(descriptor_proto));
  }

  for (const auto& [operand_id, operations] : operand_to_dependent_operations) {
    services::webnn::proto::OperationIds operations_ids_proto;
    for (size_t operation_id : operations) {
      operations_ids_proto.add_id(operation_id);
    }
    operand_to_dependent_operations_proto->emplace(
        operand_id, std::move(operations_ids_proto));
  }

  return resource_info_proto.SerializeToString(&str);
}

// static
WebNNGraphImpl::ComputeResourceInfo
WebNNGraphImpl::ComputeResourceInfo::ParseFromString(std::string_view str) {
  services::webnn::proto::ComputeResourceInfo resource_info_proto;
  CHECK(resource_info_proto.ParseFromString(str));

  auto convert_data_type =
      [](services::webnn::proto::OperandDataType data_type) {
        switch (data_type) {
          case services::webnn::proto::OperandDataType::kFloat32:
            return OperandDataType::kFloat32;
          case services::webnn::proto::OperandDataType::kFloat16:
            return OperandDataType::kFloat16;
          case services::webnn::proto::OperandDataType::kInt32:
            return OperandDataType::kInt32;
          case services::webnn::proto::OperandDataType::kUint32:
            return OperandDataType::kUint32;
          case services::webnn::proto::OperandDataType::kInt64:
            return OperandDataType::kInt64;
          case services::webnn::proto::OperandDataType::kUint64:
            return OperandDataType::kUint64;
          case services::webnn::proto::OperandDataType::kInt8:
            return OperandDataType::kInt8;
          case services::webnn::proto::OperandDataType::kUint8:
            return OperandDataType::kUint8;
          case services::webnn::proto::OperandDataType::kInt4:
            return OperandDataType::kInt4;
          case services::webnn::proto::OperandDataType::kUint4:
            return OperandDataType::kUint4;
        }
      };

  base::flat_map<std::string, OperandDescriptor> input_names_to_descriptors;
  for (const auto& [name, descriptor_proto] :
       resource_info_proto.input_names_to_descriptors()) {
    std::vector<uint32_t> shape;
    for (uint32_t dim : descriptor_proto.dim()) {
      shape.push_back(dim);
    }
    auto descriptor =
        OperandDescriptor::CreateForDeserialization(
            convert_data_type(descriptor_proto.data_type()), shape)
            .value();
    input_names_to_descriptors.emplace(name, std::move(descriptor));
  }

  base::flat_map<std::string, OperandDescriptor> output_names_to_descriptors;
  for (const auto& [name, descriptor_proto] :
       resource_info_proto.output_names_to_descriptors()) {
    std::vector<uint32_t> shape;
    for (uint32_t dim : descriptor_proto.dim()) {
      shape.push_back(dim);
    }
    auto descriptor =
        OperandDescriptor::CreateForDeserialization(
            convert_data_type(descriptor_proto.data_type()), shape)
            .value();
    output_names_to_descriptors.emplace(name, std::move(descriptor));
  }

  base::flat_map<uint64_t, base::flat_set<size_t>>
      operand_to_dependent_operations;
  for (const auto& [operand_id, operations_proto] :
       resource_info_proto.operand_to_dependent_operations()) {
    base::flat_set<size_t> operation_ids;
    for (uint64_t operation_id : operations_proto.id()) {
      operation_ids.insert(base::checked_cast<size_t>(operation_id));
    }
    operand_to_dependent_operations.emplace(operand_id,
                                            std::move(operation_ids));
  }

  return {std::move(input_names_to_descriptors),
          std::move(output_names_to_descriptors),
          std::move(operand_to_dependent_operations)};
}

WebNNGraphImpl::WebNNGraphImpl(
    mojo::PendingAssociatedReceiver<mojom::WebNNGraph> receiver,
    WebNNContextImpl* context,
    ComputeResourceInfo compute_resource_info)
    : compute_resource_info_(std::move(compute_resource_info)),
      context_(context),
      receiver_(this, std::move(receiver)) {
  CHECK(context_);
#if DCHECK_IS_ON()
  context_->AssertCalledOnValidSequence();
#endif
  // Safe to use base::Unretained because `this` owns `receiver_`.
  receiver_.set_disconnect_handler(base::BindOnce(
      &WebNNGraphImpl::OnConnectionError, base::Unretained(this)));
}

WebNNGraphImpl::~WebNNGraphImpl() = default;

void WebNNGraphImpl::OnConnectionError() {
  context_->DisconnectAndDestroyWebNNGraphImpl(handle());
}

void WebNNGraphImpl::Dispatch(
    const base::flat_map<std::string, blink::WebNNTensorToken>& named_inputs,
    const base::flat_map<std::string, blink::WebNNTensorToken>& named_outputs) {
  if (!ValidateWebNNTensorsUsage(named_inputs, named_outputs)) {
    receiver_.ReportBadMessage(kBadMessageInvalidTensor);
    return;
  }

  // Resolve the token of a input MLTensor to the corresponding `WebNNTensor`
  // instance.
  std::vector<std::pair<std::string_view, WebNNTensorImpl*>>
      name_to_input_tensors;
  name_to_input_tensors.reserve(named_inputs.size());
  for (const auto& [name, tensor_handle] : named_inputs) {
    base::optional_ref<WebNNTensorImpl> input_tensor =
        context_->GetWebNNTensorImpl(tensor_handle);
    if (!input_tensor.has_value()) {
      return;
    }
    name_to_input_tensors.emplace_back(name, input_tensor.as_ptr());
  }
  base::flat_map<std::string_view, WebNNTensorImpl*> name_to_input_tensor_map(
      std::move(name_to_input_tensors));
  if (!ValidateWebNNTensors(
          name_to_input_tensor_map,
          compute_resource_info_.input_names_to_descriptors)) {
    receiver_.ReportBadMessage(kBadMessageInvalidTensor);
    return;
  }

  // Resolve the token of a output MLTensor to the corresponding `WebNNTensor`
  // instance.
  std::vector<std::pair<std::string_view, WebNNTensorImpl*>>
      name_to_output_tensors;
  name_to_output_tensors.reserve(named_outputs.size());
  for (const auto& [name, tensor_handle] : named_outputs) {
    base::optional_ref<WebNNTensorImpl> output_tensor =
        context_->GetWebNNTensorImpl(tensor_handle);
    if (!output_tensor.has_value()) {
      return;
    }
    name_to_output_tensors.emplace_back(name, output_tensor.as_ptr());
  }

  base::flat_map<std::string_view, WebNNTensorImpl*> name_to_output_tensor_map(
      std::move(name_to_output_tensors));
  if (!ValidateWebNNTensors(
          name_to_output_tensor_map,
          compute_resource_info_.output_names_to_descriptors)) {
    receiver_.ReportBadMessage(kBadMessageInvalidTensor);
    return;
  }

  // Call DispatchImpl() implemented by an `mojom::WebNNGraph` backend.
  DispatchImpl(name_to_input_tensor_map, name_to_output_tensor_map);
}

void WebNNGraphImpl::SaveGraph(const std::string& key) {
  SaveGraphImpl(key);
}

}  // namespace webnn
