#pragma once

#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <variant>
#include <vector>

using metavalue = std::variant<
    uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t,
    float, double, bool, std::string, std::vector<uint8_t>, std::vector<int8_t>,
    std::vector<uint16_t>, std::vector<int16_t>, std::vector<uint32_t>,
    std::vector<int32_t>, std::vector<uint64_t>, std::vector<int64_t>,
    std::vector<float>, std::vector<double>, std::vector<bool>,
    std::vector<std::string>>;

class GGUF {
public:
  char *pth;
  int fd = -1;
  size_t fsize = 0;
  void *data = nullptr;
  size_t cursor = 0;

  uint32_t magic;
  uint32_t version;
  uint64_t tensor_count;
  uint64_t metadata_count;
  enum ggml_type : uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_BF16 = 30,
    GGML_TYPE_TQ1_0 = 34,
    GGML_TYPE_TQ2_0 = 35,
    GGML_TYPE_MXFP4 = 39,
    GGML_TYPE_COUNT = 40,
  };

  enum gguf_metadata_value_type : uint32_t {
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
  };

  struct Metadata {
    std::string key;
    uint32_t type;
    metavalue value;
  };

  struct Tensor {
    std::string name;
    std::vector<int64_t> dimensions;
    ggml_type type;
    uint64_t offset;
    size_t nbytes = 0;
    const void *data = nullptr;
  };

  std::vector<Metadata> metadata_entries;
  std::vector<Tensor> tensor_entries;
  uint32_t alignment = 32;
  size_t tensor_data_offset = 0;

  GGUF(char *pth);
  ~GGUF();
  template <typename T> T get_data_as() {
    T value = *reinterpret_cast<T *>(static_cast<char *>(data) + cursor);
    cursor += sizeof(T);
    return value;
  }
  std::string readString() {
    uint64_t len = get_data_as<uint64_t>();
    std::string str(
        reinterpret_cast<char *>(static_cast<char *>(data) + cursor), len);
    cursor += len;
    return str;
  }
  metavalue readArrayAsValue(uint32_t elem_type, uint64_t n) {
    switch (elem_type) {
    case GGUF_METADATA_VALUE_TYPE_UINT8: {
      std::vector<uint8_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<uint8_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_INT8: {
      std::vector<int8_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<int8_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_UINT16: {
      std::vector<uint16_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<uint16_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_INT16: {
      std::vector<int16_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<int16_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_UINT32: {
      std::vector<uint32_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<uint32_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_INT32: {
      std::vector<int32_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<int32_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_FLOAT32: {
      std::vector<float> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<float>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_BOOL: {
      std::vector<bool> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(static_cast<bool>(get_data_as<uint8_t>()));
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_STRING: {
      std::vector<std::string> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(readString());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_UINT64: {
      std::vector<uint64_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<uint64_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_INT64: {
      std::vector<int64_t> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<int64_t>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
      std::vector<double> values;
      values.reserve(static_cast<size_t>(n));
      for (uint64_t i = 0; i < n; i++) {
        values.push_back(get_data_as<double>());
      }
      return values;
    }
    case GGUF_METADATA_VALUE_TYPE_ARRAY:
      throw std::runtime_error("nested array metadata not supported");
    default:
      throw std::runtime_error("invalid array element type");
    }
  }
  metavalue readMetadataValue(uint32_t type) {
    switch (type) {
    case GGUF_METADATA_VALUE_TYPE_UINT8:
      return get_data_as<uint8_t>();
    case GGUF_METADATA_VALUE_TYPE_INT8:
      return get_data_as<int8_t>();
    case GGUF_METADATA_VALUE_TYPE_UINT16:
      return get_data_as<uint16_t>();
    case GGUF_METADATA_VALUE_TYPE_INT16:
      return get_data_as<int16_t>();
    case GGUF_METADATA_VALUE_TYPE_UINT32:
      return get_data_as<uint32_t>();
    case GGUF_METADATA_VALUE_TYPE_INT32:
      return get_data_as<int32_t>();
    case GGUF_METADATA_VALUE_TYPE_FLOAT32:
      return get_data_as<float>();
    case GGUF_METADATA_VALUE_TYPE_BOOL:
      return static_cast<bool>(get_data_as<uint8_t>());
    case GGUF_METADATA_VALUE_TYPE_STRING:
      return readString();
    case GGUF_METADATA_VALUE_TYPE_ARRAY: {
      uint32_t elem_type = get_data_as<uint32_t>();
      uint64_t n = get_data_as<uint64_t>();
      return readArrayAsValue(elem_type, n);
    }
    case GGUF_METADATA_VALUE_TYPE_UINT64:
      return get_data_as<uint64_t>();
    case GGUF_METADATA_VALUE_TYPE_INT64:
      return get_data_as<int64_t>();
    case GGUF_METADATA_VALUE_TYPE_FLOAT64:
      return get_data_as<double>();
    default:
      throw std::runtime_error("invalid metadata value type");
    }
  }
  void readMetadata();
  void readTensors();
  void readTensorData();
  const Tensor *findTensor(const std::string &name) const;
  const void *getTensorData(const std::string &name) const;
  uint32_t getAlignment() const;
  size_t tensorByteSize(const Tensor &tensor) const;
  const char *typeName(ggml_type type) const;
  void printMetadata() const;
  void printTensors() const;
  void printTensorData(const std::string &name, size_t preview_count = 8) const;
  void printHeader();
};
