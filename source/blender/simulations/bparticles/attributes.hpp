#pragma once

#include <string>

#include "BLI_array_ref.hpp"
#include "BLI_vector.hpp"
#include "BLI_math.hpp"
#include "BLI_string_ref.hpp"
#include "BLI_range.hpp"
#include "BLI_set_vector.hpp"
#include "BLI_array_allocator.hpp"
#include "BLI_optional.hpp"

namespace BParticles {

using BLI::ArrayAllocator;
using BLI::ArrayRef;
using BLI::float2;
using BLI::float3;
using BLI::Optional;
using BLI::Range;
using BLI::rgba_b;
using BLI::rgba_f;
using BLI::SetVector;
using BLI::StringRef;
using BLI::StringRefNull;
using BLI::Vector;

/**
 * Possible types of attributes. All types are expected to be POD (plain old data).
 * New types can be added when necessary.
 */
enum AttributeType {
  Byte,
  Integer,
  Float,
  Float2,
  Float3,
  RGBA_b,
  RGBA_f,
};

template<typename T> struct attribute_type_by_type {
};

#define ATTRIBUTE_TYPE_BY_TYPE(CPP_TYPE, ATTRIBUTE_TYPE) \
  template<> struct attribute_type_by_type<CPP_TYPE> { \
    static const AttributeType value = AttributeType::ATTRIBUTE_TYPE; \
  }

ATTRIBUTE_TYPE_BY_TYPE(uint8_t, Byte);
ATTRIBUTE_TYPE_BY_TYPE(int32_t, Integer);
ATTRIBUTE_TYPE_BY_TYPE(float, Float);
ATTRIBUTE_TYPE_BY_TYPE(float2, Float2);
ATTRIBUTE_TYPE_BY_TYPE(float3, Float3);
ATTRIBUTE_TYPE_BY_TYPE(rgba_b, RGBA_b);
ATTRIBUTE_TYPE_BY_TYPE(rgba_f, RGBA_f);

#undef ATTRIBUTE_TYPE_BY_TYPE

/**
 * Get the size of an attribute type.
 *
 * TODO(jacques): Figure out how to make type.size() work nicely instead.
 */
inline uint size_of_attribute_type(AttributeType type)
{
  switch (type) {
    case AttributeType::Byte:
      return sizeof(uint8_t);
    case AttributeType::Integer:
      return sizeof(int32_t);
    case AttributeType::Float:
      return sizeof(float);
    case AttributeType::Float2:
      return sizeof(float2);
    case AttributeType::Float3:
      return sizeof(float3);
    case AttributeType::RGBA_b:
      return sizeof(rgba_b);
    case AttributeType::RGBA_f:
      return sizeof(rgba_f);
  };
  BLI_assert(false);
  return 0;
}

#define MAX_ATTRIBUTE_SIZE sizeof(rgba_f)

struct AnyAttributeValue {
  char storage[MAX_ATTRIBUTE_SIZE];

  template<typename T> static AnyAttributeValue FromValue(T value)
  {
    BLI_STATIC_ASSERT(attribute_type_by_type<T>::value >= 0, "");
    BLI_STATIC_ASSERT(sizeof(T) <= MAX_ATTRIBUTE_SIZE, "");
    AnyAttributeValue attribute;
    memcpy(attribute.storage, &value, sizeof(T));
    return attribute;
  }
};

class AttributesInfo;

class AttributesDeclaration {
 private:
  SetVector<std::string> m_names;
  Vector<AttributeType> m_types;
  Vector<AnyAttributeValue> m_defaults;

  friend AttributesInfo;

 public:
  AttributesDeclaration() = default;

  template<typename T> void add(StringRef name, T default_value)
  {
    if (m_names.add(name.to_std_string())) {
      AttributeType type = attribute_type_by_type<T>::value;
      m_types.append(type);
      m_defaults.append(AnyAttributeValue::FromValue(default_value));
    }
  }

  uint size() const
  {
    return m_names.size();
  }

  void join(AttributesDeclaration &other);
  void join(AttributesInfo &other);
};

/**
 * Contains information about a set of attributes. Every attribute is identified by a unique name
 * and a unique index. So two attributes of different types have to have different names.
 *
 * Furthermore, every attribute has a default value.
 */
class AttributesInfo {
 private:
  SetVector<std::string> m_names;
  Vector<AttributeType> m_types;
  Vector<AnyAttributeValue> m_defaults;

  friend AttributesDeclaration;

 public:
  AttributesInfo() = default;
  AttributesInfo(AttributesDeclaration &builder);

  /**
   * Get the number of different attributes.
   */
  uint size() const
  {
    return m_names.size();
  }

  /**
   * Get the attribute name that corresponds to an index.
   * Asserts when the index is too large.
   */
  StringRefNull name_of(uint index) const
  {
    return m_names[index];
  }

  /**
   * Get the type of an attribute identified by its index.
   * Asserts when the index is too large.
   */
  AttributeType type_of(uint index) const
  {
    return m_types[index];
  }

  /**
   * Get the type of an attribute identified by its name.
   * Asserts when the name does not exist.
   */
  AttributeType type_of(StringRef name) const
  {
    return this->type_of(this->attribute_index(name));
  }

  /**
   * Get the types of all attributes. The index into the array is the index of the corresponding
   * attribute.
   */
  ArrayRef<AttributeType> types() const
  {
    return m_types;
  }

  /**
   * Get the index corresponding to an attribute name.
   * Returns -1 when the attribute does not exist.
   */
  int attribute_index_try(StringRef name) const
  {
    return m_names.index(name.to_std_string());
  }

  /**
   * Get the index corresponding to an attribute with the given name and type.
   * Returns -1 when the attribute does not exist.
   */
  int attribute_index_try(StringRef name, AttributeType type) const
  {
    int index = this->attribute_index_try(name);
    if (index == -1) {
      return -1;
    }
    else if (this->type_of((uint)index) == type) {
      return index;
    }
    else {
      return -1;
    }
  }

  /**
   * Get the index corresponding to an attribute name.
   * Asserts when the attribute does not exist.
   */
  uint attribute_index(StringRef name) const
  {
    int index = this->attribute_index_try(name);
    BLI_assert(index >= 0);
    return (uint)index;
  }

  /**
   * Get a range with all attribute indices.
   * The range will start at 0.
   */
  Range<uint> attribute_indices() const
  {
    return Range<uint>(0, this->size());
  }

  /**
   * Get a pointer to the default value of an attribute.
   */
  void *default_value_ptr(uint index) const
  {
    BLI_assert(index < this->size());
    return (void *)m_defaults[index].storage;
  }

  /**
   * Don't do a deep comparison for now. This might change later.
   */
  friend bool operator==(const AttributesInfo &a, const AttributesInfo &b)
  {
    return &a == &b;
  }
};

class AttributeArrays;

/**
 * Contains a memory buffer for every attribute in an AttributesInfo object.
 * All buffers have equal element-length but not necessarily equal byte-length.
 *
 * The pointers are not owned by this structure. They are passed on creation and have to be freed
 * manually. This is necessary because in different contexts, it makes sense to allocate the
 * buffers in different ways. Nevertheless, there are some utilities to simplify allocation and
 * deallocation in common cases.
 *
 * Most code does not use this class directly. Instead it uses AttributeArrays, which is just a
 * slice of this.
 */
class AttributeArraysCore {
 private:
  AttributesInfo *m_info;
  Vector<void *> m_arrays;
  uint m_size = 0;

 public:
  AttributeArraysCore(AttributesInfo &info, ArrayRef<void *> arrays, uint size);
  ~AttributeArraysCore();

  /**
   * Create a new instance in which the pointers are all separately allocated using MEM_mallocN.
   */
  static AttributeArraysCore NewWithSeparateAllocations(AttributesInfo &info, uint size);
  /**
   * Free all buffers separately using MEM_freeN.
   */
  void free_buffers();

  /**
   * Create a new instance in which all pointers are separately allocated from a
   * fixed-array-allocator. No separate length has to be provided, since the allocator only
   * allocates arrays of one specific length.
   */
  static AttributeArraysCore NewWithArrayAllocator(AttributesInfo &info,
                                                   ArrayAllocator &allocator);

  /**
   * Deallocate pointers in the given fixed-array-allocator.
   */
  void deallocate_in_array_allocator(ArrayAllocator &allocator);

  /**
   * Get information about the stored attributes.
   */
  AttributesInfo &info();

  /**
   * Get the raw pointer to the beginning of an attribute array identified by an index.
   */
  void *get_ptr(uint index);

  /**
   * Get the type of an attribute identified by an index.
   */
  AttributeType get_type(uint index);

  /**
   * Get a slice containing everything for further processing.
   */
  AttributeArrays slice_all();

  /**
   * Get the number of elements stored per attribute.
   */
  uint size() const;

  /**
   * Get all raw pointers.
   */
  ArrayRef<void *> pointers();
};

/**
 * The main class used to interact with attributes. It represents a continuous slice of an
 * AttributeArraysCore instance. So, it is very light weight and can be passed by value.
 */
class AttributeArrays {
 private:
  AttributeArraysCore &m_core;
  uint m_start, m_size;

 public:
  AttributeArrays(AttributeArraysCore &core, uint start, uint size);

  /**
   * Get the number of referenced elements.
   */
  uint size() const;

  /**
   * Get information about the referenced attributes.
   */
  AttributesInfo &info();

  /**
   * Get the index of an attributed identified by a name.
   */
  uint attribute_index(StringRef name);

  /**
   * Get the size of an element in one attribute.
   */
  uint attribute_stride(uint index);

  /**
   * Get the raw pointer to the buffer that contains attribute values.
   */
  void *get_ptr(uint index) const;

  /**
   * Initialize an attribute array using its default value.
   */
  void init_default(uint index);
  void init_default(StringRef name);

  /**
   * Get access to the underlying attribute arrays.
   * Asserts when the attribute does not exists.
   */
  template<typename T> ArrayRef<T> get(uint index) const
  {
    BLI_assert(attribute_type_by_type<T>::value == m_core.info().type_of(index));
    void *ptr = this->get_ptr(index);
    return ArrayRef<T>((T *)ptr, m_size);
  }
  template<typename T> ArrayRef<T> get(StringRef name)
  {
    uint index = this->attribute_index(name);
    return this->get<T>(index);
  }

  /**
   * Get access to the arrays.
   * Does not assert when the attribute does not exist.
   */
  template<typename T> Optional<ArrayRef<T>> try_get(StringRef name)
  {
    int index = this->info().attribute_index_try(name, attribute_type_by_type<T>::value);
    if (index == -1) {
      return {};
    }
    else {
      return this->get<T>((uint)index);
    }
  }

  /**
   * Get a continuous slice of the attribute arrays.
   */
  AttributeArrays slice(uint start, uint size) const;

  /**
   * Create a new slice containing only the first n elements.
   */
  AttributeArrays take_front(uint n) const;
};

/* Attribute Arrays Core
 *****************************************/

inline AttributesInfo &AttributeArraysCore::info()
{
  return *m_info;
}

inline void *AttributeArraysCore::get_ptr(uint index)
{
  return m_arrays[index];
}

inline AttributeType AttributeArraysCore::get_type(uint index)
{
  return m_info->type_of(index);
}

inline AttributeArrays AttributeArraysCore::slice_all()
{
  return AttributeArrays(*this, 0, m_size);
}

inline uint AttributeArraysCore::size() const
{
  return m_size;
}

inline ArrayRef<void *> AttributeArraysCore::pointers()
{
  return m_arrays;
}

/* Attribute Arrays
 ******************************************/

inline AttributeArrays::AttributeArrays(AttributeArraysCore &core, uint start, uint size)
    : m_core(core), m_start(start), m_size(size)
{
  BLI_assert(m_start + m_size <= m_core.size());
}

inline uint AttributeArrays::size() const
{
  return m_size;
}

inline AttributesInfo &AttributeArrays::info()
{
  return m_core.info();
}

inline uint AttributeArrays::attribute_index(StringRef name)
{
  return this->info().attribute_index(name);
}

inline uint AttributeArrays::attribute_stride(uint index)
{
  return size_of_attribute_type(this->info().type_of(index));
}

inline void *AttributeArrays::get_ptr(uint index) const
{
  void *ptr = m_core.get_ptr(index);
  AttributeType type = m_core.get_type(index);
  uint size = size_of_attribute_type(type);
  return POINTER_OFFSET(ptr, m_start * size);
}

inline void AttributeArrays::init_default(uint index)
{
  void *default_value = m_core.info().default_value_ptr(index);
  void *dst = this->get_ptr(index);
  AttributeType type = m_core.get_type(index);
  uint element_size = size_of_attribute_type(type);

  for (uint i = 0; i < m_size; i++) {
    memcpy(POINTER_OFFSET(dst, element_size * i), default_value, element_size);
  }
}

inline void AttributeArrays::init_default(StringRef name)
{
  this->init_default(this->attribute_index(name));
}

inline AttributeArrays AttributeArrays::slice(uint start, uint size) const
{
  return AttributeArrays(m_core, m_start + start, size);
}

inline AttributeArrays AttributeArrays::take_front(uint n) const
{
  BLI_assert(n <= m_size);
  return AttributeArrays(m_core, m_start, n);
}

}  // namespace BParticles
