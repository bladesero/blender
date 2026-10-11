/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "gpu_shader_ray_query_lib.glsl"

struct RayQueryTest {
  [[acceleration_structure(0)]] accelerationStructureEXT scene;
  [[storage(1, write)]] float4 (&result)[];
  [[storage(2, read)]] const float4 (&ray)[];
};

[[compute, local_size(1)]]
void ray_query_test([[resource_table]] RayQueryTest &srt)
{
  GPURayQueryHit hit = gpu_ray_query(
      srt.scene, srt.ray[0].xyz, srt.ray[1].xyz, srt.ray[0].w, srt.ray[1].w);
  srt.result[0] = float4(
      hit.distance, float(hit.instance), float(hit.primitive), float(hit.valid));
  srt.result[1] = float4(hit.barycentric, float(hit.front_face), 0.0f);
}

PipelineCompute gpu_ray_query_test(ray_query_test);

[[compute, local_size(1)]]
void ray_query_candidate_test([[resource_table]] RayQueryTest &srt)
{
  srt.result[0] = float4(0);
  srt.result[1] = float4(0);
#ifdef GPU_RAY_QUERY
  rayQueryEXT query;
  rayQueryInitializeEXT(
      query, srt.scene, 0u, 0xffu, srt.ray[0].xyz, srt.ray[0].w, srt.ray[1].xyz, srt.ray[1].w);
  while (rayQueryProceedEXT(query)) {
    GPURayQueryHit candidate = gpu_ray_query_candidate(query);
    if (candidate.valid) {
      /* Simulate a transparent foreground by rejecting primitive zero. */
      srt.result[1].w += 1.0f;
      if (candidate.primitive != 0u) {
        srt.result[1].xyz = float3(candidate.barycentric, float(candidate.instance));
        rayQueryConfirmIntersectionEXT(query);
      }
    }
  }
  if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
    srt.result[0] = float4(rayQueryGetIntersectionTEXT(query, true),
                           float(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)),
                           float(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)),
                           1.0f);
  }
#endif
}

PipelineCompute gpu_ray_query_candidate_test(ray_query_candidate_test);
