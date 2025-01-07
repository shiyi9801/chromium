// Copyright 2024 The Chromium Authors
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#ifndef SERVICES_WEBNN_ORT_UTILS_ORT_H_
#define SERVICES_WEBNN_ORT_UTILS_ORT_H_

#include "services/webnn/public/cpp/operand_descriptor.h"
#include "services/webnn/public/mojom/webnn_error.mojom.h"
#include "third_party/onnxruntime_headers/src/include/onnxruntime/core/session/onnxruntime_c_api.h"

namespace webnn::ort {

class Float16 {
 public:
  Float16();

  static Float16 FromFloat32(float f32);

  float ToFloat32() const;

 private:
  explicit Float16(uint16_t raw_bits);

  uint16_t bits_;
};

ONNXTensorElementDataType OperandTypeToONNXTensorElementDataType(
    OperandDataType data_type);

const OrtApi* GetOrtApi();

const OrtModelBuilderApi* GetOrtModelBuilderApi();

mojom::ErrorPtr CreateError(mojom::Error::Code error_code,
                            const std::string& error_message,
                            std::string_view label = "");

}  // namespace webnn::ort

#endif  // SERVICES_WEBNN_ORT_UTILS_ORT_H_
