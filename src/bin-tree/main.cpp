#include <array>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <vector>
#include <limits>

typedef union
{
  char bytes[4];
  uint32_t uint32;
  int32_t int32;
  float value;
} Data32;

struct float32_compare
{
  bool operator()(float a, float b) const
  {
    static const auto epsilon = std::numeric_limits<float>::epsilon();
    return b - a < epsilon;
  }
};

typedef struct
{
  std::set<float, float32_compare> set;
  std::vector<decltype(set)::iterator> array;

  decltype(array)::iterator insert(float value)
  {
    static const auto epsilon = std::numeric_limits<float>::epsilon();
    decltype(array)::size_type index_min = 0;
    auto index_max = array.size();
    while (index_max > index_min)
    {
      auto index_mid = (index_max + index_min) / 2;
      auto value_mid = *array[index_mid];
      auto diff_mid = value - value_mid;
      if (diff_mid <= -epsilon)
      {
        index_max = index_mid;
      }
      else if (diff_mid >= epsilon)
      {
        index_min = index_mid + 1;
      }
      else
      {
        index_min = index_mid;
        break;
      }
    }
    auto location = array.begin();
    std::advance(location, index_min);
    auto insert_result = set.insert(value);
    if (insert_result.second)
    {
      array.insert(location, insert_result.first);
    }
    return location;
  }
} OrderedFloatSet;

int main()
{
  std::cin.exceptions(std::ios::failbit | std::ios::badbit | std::ios::eofbit);
  Data32 vertex_count, normal_count, texcoord_count, face_count;
  std::cin.read(vertex_count.bytes, sizeof(vertex_count));
  std::cin.read(normal_count.bytes, sizeof(vertex_count));
  std::cin.read(texcoord_count.bytes, sizeof(vertex_count));
  std::cin.read(face_count.bytes, sizeof(vertex_count));

  auto source_vertex_buffer = std::make_unique<std::array<float, 3>[]>(vertex_count.uint32 * sizeof(float));
  auto source_normal_buffer = std::make_unique<std::array<float, 3>[]>(normal_count.uint32 * sizeof(float));
  auto source_texcoord_buffer = std::make_unique<std::array<float, 2>[]>(texcoord_count.uint32 * sizeof(float));
  auto source_face_buffer = std::make_unique<std::array<std::array<uint32_t, 3>, 3>[]>(face_count.uint32 * sizeof(float));
  return 0;
}