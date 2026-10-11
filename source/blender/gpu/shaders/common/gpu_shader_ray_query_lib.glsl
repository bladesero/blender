/* SPDX-FileCopyrightText: 2026 Blender Authors
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#pragma once
#include "gpu_shader_compat.hh"

struct GPURayQueryHit {
  bool valid;
  bool front_face;
  float distance;
  uint instance;
  uint primitive;
  float2 barycentric;
};

GPURayQueryHit gpu_ray_query(accelerationStructureEXT scene,
                             float3 origin,
                             float3 direction,
                             float min_distance,
                             float max_distance)
{
  GPURayQueryHit hit;
  hit.valid = false;
  hit.front_face = true;
  hit.distance = max_distance;
  hit.instance = 0u;
  hit.primitive = 0u;
  hit.barycentric = float2(0.0f);
#ifdef GPU_RAY_QUERY
  rayQueryEXT query;
  rayQueryInitializeEXT(
      query, scene, gl_RayFlagsOpaqueEXT, 0xffu, origin, min_distance, direction, max_distance);
  while (rayQueryProceedEXT(query)) {
  }
  hit.valid = rayQueryGetIntersectionTypeEXT(query, true) ==
              gl_RayQueryCommittedIntersectionTriangleEXT;
  if (hit.valid) {
    hit.distance = rayQueryGetIntersectionTEXT(query, true);
    hit.instance = rayQueryGetIntersectionInstanceCustomIndexEXT(query, true);
    hit.primitive = rayQueryGetIntersectionPrimitiveIndexEXT(query, true);
    hit.barycentric = rayQueryGetIntersectionBarycentricsEXT(query, true);
    hit.front_face = rayQueryGetIntersectionFrontFaceEXT(query, true);
  }
#endif
  return hit;
}

#ifdef GPU_RAY_QUERY
/** Read a non-opaque candidate without committing it. Material evaluation decides acceptance. */
GPURayQueryHit gpu_ray_query_candidate(rayQueryEXT query)
{
  GPURayQueryHit hit;
  hit.valid = rayQueryGetIntersectionTypeEXT(query, false) ==
              gl_RayQueryCandidateIntersectionTriangleEXT;
  hit.front_face = true;
  hit.distance = 0.0f;
  hit.instance = hit.primitive = 0u;
  hit.barycentric = float2(0.0f);
  if (hit.valid) {
    hit.distance = rayQueryGetIntersectionTEXT(query, false);
    hit.instance = rayQueryGetIntersectionInstanceCustomIndexEXT(query, false);
    hit.primitive = rayQueryGetIntersectionPrimitiveIndexEXT(query, false);
    hit.barycentric = rayQueryGetIntersectionBarycentricsEXT(query, false);
    hit.front_face = rayQueryGetIntersectionFrontFaceEXT(query, false);
  }
  return hit;
}
#endif
