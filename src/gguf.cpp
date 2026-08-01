#include "gguf.h"

#include "ggml.h"

#include <algorithm>
#include <sstream>

namespace {

size_t padOffset(size_t offset, size_t alignment) {
  return (offset + alignment - 1) & ~(alignment - 1);
}

} // namespace

GGUF::GGUF(char *pth) : pth(pth) {
  fd = open(pth, O_RDONLY);
  if (fd < 0) {
    throw std::runtime_error(std::string("failed to open file: ") + pth);
  }

  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    throw std::runtime_error(std::string("failed to stat file: ") + pth);
  }

  fsize = static_cast<size_t>(st.st_size);
  data = mmap(nullptr, fsize, PROT_READ, MAP_PRIVATE, fd, 0);
  if (data == MAP_FAILED) {
    close(fd);
    throw std::runtime_error(std::string("failed to mmap file: ") + pth);
  }
  magic = get_data_as<uint32_t>();
  version = get_data_as<uint32_t>();
  tensor_count = get_data_as<uint64_t>();
  metadata_count = get_data_as<uint64_t>();
  readMetadata();
  readTensors();
  readTensorData();
}

GGUF::~GGUF() {
  if (data != nullptr && data != MAP_FAILED) {
    munmap(data, fsize);
  }
  if (fd >= 0) {
    close(fd);
  }
}

void GGUF::readMetadata() {
  for (uint64_t i = 0; i < metadata_count; i++) {
    std::string key = readString();
    uint32_t type = get_data_as<uint32_t>();
    metavalue value = readMetadataValue(type);
    metadata_entries.push_back({key, type, value});
  }
}

uint32_t GGUF::getAlignment() const {
  for (const auto &entry : metadata_entries) {
    if (entry.key == "general.alignment") {
      if (const auto *value = std::get_if<uint32_t>(&entry.value)) {
        return *value;
      }
    }
  }
  return 32;
}

const char *GGUF::typeName(ggml_type type) const {
  return ggml_type_name(static_cast<enum ::ggml_type>(type));
}

size_t GGUF::tensorByteSize(const Tensor &tensor) const {
  int64_t ne[GGML_MAX_DIMS] = {1, 1, 1, 1};
  for (size_t i = 0; i < tensor.dimensions.size() && i < GGML_MAX_DIMS; ++i) {
    ne[i] = tensor.dimensions[i];
  }

  const auto type = static_cast<enum ::ggml_type>(tensor.type);
  const int64_t blck_size = ggml_blck_size(type);
  if (blck_size == 0 || ne[0] % blck_size != 0) {
    throw std::runtime_error(std::string("invalid tensor shape for type ") +
                             tensor.name);
  }

  int64_t nblocks = 1;
  for (int i = 0; i < GGML_MAX_DIMS; ++i) {
    nblocks *= ne[i];
  }
  nblocks /= blck_size;

  return static_cast<size_t>(nblocks) * ggml_type_size(type);
}

void GGUF::readTensors() {
  tensor_entries.clear();
  tensor_entries.reserve(static_cast<size_t>(tensor_count));

  for (uint64_t i = 0; i < tensor_count; ++i) {
    Tensor tensor;
    tensor.name = readString();
    const uint32_t n_dims = get_data_as<uint32_t>();
    tensor.dimensions.resize(n_dims);
    for (uint32_t d = 0; d < n_dims; ++d) {
      tensor.dimensions[d] = get_data_as<int64_t>();
    }
    tensor.type = static_cast<ggml_type>(get_data_as<uint32_t>());
    tensor.offset = get_data_as<uint64_t>();
    tensor.nbytes = tensorByteSize(tensor);
    tensor_entries.push_back(std::move(tensor));
  }

  alignment = getAlignment();
  if (tensor_count > 0) {
    tensor_data_offset = padOffset(cursor, alignment);
  }
}

const GGUF::Tensor *GGUF::findTensor(const std::string &name) const {
  for (const auto &tensor : tensor_entries) {
    if (tensor.name == name) {
      return &tensor;
    }
  }
  return nullptr;
}

const void *GGUF::getTensorData(const std::string &name) const {
  const Tensor *tensor = findTensor(name);
  return tensor != nullptr ? tensor->data : nullptr;
}

void GGUF::readTensorData() {
  if (tensor_count == 0) {
    return;
  }

  for (auto &tensor : tensor_entries) {
    const size_t file_offset = tensor_data_offset + tensor.offset;
    if (file_offset > fsize || tensor.nbytes > fsize - file_offset) {
      throw std::runtime_error("tensor data out of bounds: " + tensor.name);
    }
    tensor.data = static_cast<const char *>(data) + file_offset;
  }
}

void GGUF::printTensorData(const std::string &name,
                           size_t preview_count) const {
  const Tensor *tensor = findTensor(name);
  if (tensor == nullptr) {
    throw std::runtime_error("tensor not found: " + name);
  }
  if (tensor->data == nullptr) {
    throw std::runtime_error("tensor data not loaded: " + name);
  }

  std::cout << tensor->name << " data preview:" << std::endl;

  if (tensor->type == GGML_TYPE_F32) {
    const auto *values = static_cast<const float *>(tensor->data);
    const size_t count =
        std::min(preview_count, tensor->nbytes / sizeof(float));
    for (size_t i = 0; i < count; ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << values[i];
    }
    std::cout << std::endl;
    return;
  }

  const auto *bytes = static_cast<const uint8_t *>(tensor->data);
  const size_t count = std::min(preview_count, tensor->nbytes);
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      std::cout << " ";
    }
    std::cout << std::hex << static_cast<int>(bytes[i]) << std::dec;
  }
  std::cout << std::endl;
}

void GGUF::printMetadata() const {
  for (const auto &entry : metadata_entries) {
    std::cout << entry.key << std::endl;
  }
}

void GGUF::printTensors() const {
  for (const auto &tensor : tensor_entries) {
    std::cout << tensor.name << " | " << typeName(tensor.type) << " | [";
    for (size_t i = 0; i < tensor.dimensions.size(); ++i) {
      if (i > 0) {
        std::cout << ", ";
      }
      std::cout << tensor.dimensions[i];
    }
    std::cout << "] | offset=" << tensor.offset << " | " << tensor.nbytes
              << " bytes" << std::endl;
  }
  if (tensor_count > 0) {
    std::cout << "Tensor data starts at file offset " << tensor_data_offset
              << " (alignment=" << alignment << ")" << std::endl;
  }
}

void GGUF::printHeader() {
  std::cout << "Magic: " << magic << std::endl;
  std::cout << "Version: " << version << std::endl;
  std::cout << "Tensor count: " << tensor_count << std::endl;
  std::cout << "Metadata count: " << metadata_count << std::endl;
}
