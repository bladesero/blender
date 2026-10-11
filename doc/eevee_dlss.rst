EEVEE Vulkan DLSS Reconstruction
===============================

Scope
-----

This experimental path reconstructs the Combined viewport or final-render image before
color management. It uses NVIDIA NGX 310.4.0 directly: SR transformer preset K and RR
transformer preset D. Windows, Vulkan, RTX and EEVEE are required. Cycles and frame
generation are outside its scope. Implementation and visual acceptance are separate;
the validation records below identify which configurations have actually been tested.

Build and use
-------------

On Windows x64, obtain the official NVIDIA DLSS SDK separately. SDK 310.4.0 is the
integration baseline. Its headers, static libraries and signed runtime are not vendored
in Blender's source tree. Configure with::

  cmake -S blender -B build -DWITH_VULKAN_BACKEND=ON -DWITH_NVIDIA_DLSS=ON -DNVIDIA_DLSS_ROOT=C:/SDK/DLSS
  cmake --build build --config Release --target blender
  cmake --install build --config Release

The SDK directory must contain ``include/nvsdk_ngx_helpers_vk.h``, the dynamic CRT
libraries under ``lib/Windows_x86_64/x64`` and ``lib/Windows_x86_64/rel/nvngx_dlss.dll``.
RR additionally uses ``include/nvsdk_ngx_helpers_dlssd_vk.h`` and the optional signed
``lib/Windows_x86_64/rel/nvngx_dlssd.dll``. Runtimes are copied next to the executable.
Missing RR support falls back to native denoising with SR. Debug builds use the iterator-debugging-0
variant. ``WITH_NVIDIA_DLSS`` defaults to OFF; ordinary builds do not need the SDK.

Start Blender with ``--gpu-backend vulkan``. Select EEVEE, use a rendered viewport,
then select Quality, Balanced or Performance under Render Properties > Sampling >
Viewport > DLSS Super Resolution. The corresponding RNA property is
``scene.eevee.dlss_quality``. Off restores native EEVEE. The requested quality is saved
in the blend file; support is checked locally at runtime.

Native fallback is used for unsupported GPUs/backends/builds, missing runtime, NGX
failure, non-Combined passes, render regions, overscan, XR, viewport compositing,
transparent film, and camera depth of field. The viewport information text describes
fallback. Viewport image renders retain their native path. An unsupported configuration
does not silently change the saved user setting.

Final renders use the separate ``scene.eevee.dlss_render_quality`` setting. RR switches
are ``scene.eevee.dlss_rr`` (viewport) and ``scene.eevee.dlss_render_rr`` (final). All new
settings default to off, and RR requires quality other than Off. Final output supports
Combined only; data passes, AOVs, Cryptomatte, multiview and render borders fall back.
Motion blur, DOF, volumes, transparent film, forward shading and transmission also
select native rendering. RR requires the branch's supported hardware-RT material subset;
unsupported secondary-hit materials, including normal maps and layered shaders, fall back.
Curves, point clouds, particle hair and the specialized sculpt path also reject RR.

Pipeline and lifetime
---------------------

``DLSSModule`` owns shared reconstruction and guides. ``DLSSViewportAdapter`` owns
viewport camera history and display framebuffer access; ``DLSSRenderAdapter`` owns
final linear RGBA output. Final reconstruction does not require a window or ``rv3d``.

* NGX supplies the input size for the selected output size and quality.
* EEVEE shades at input resolution. SR retains native denoising. RR traces at input
  resolution with spatial filtering, temporal and bilateral ray denoising disabled.
  The spatial resolve shader only copies one ray per pixel in this configuration.
  Lighting composition, material weights and world-space illumination caches remain.
* RR guides are captured before GBuffer and ray-time pool release. Evaluated GBuffer
  reflection color already contains integrated BRDF reflectance and closure weights.
  Normals are world-space, roughness is packed in alpha, and hit distance is in world
  units (the environment sentinel is converted to zero). RR replaces the SR call.
* A compute pass converts reverse Z to conventional depth, resolves dense camera and
  object motion, converts motion to input-pixel units, and flips inputs to top-left
  origin. Motion excludes projection jitter.
* The Vulkan external-compute graph node transitions images to GENERAL, declares all
  input/output dependencies, records NGX, and invalidates cached Vulkan bindings.
  Its memory access masks cover SDK-internal transfer clears/copies as well as shaders.
* DLSS replaces Film's color accumulation/reprojection for the active viewport.
  Its high-resolution RGB result is displayed with alpha 1. Depth for overlays is
  sampled separately; it is not reconstructed by DLSS.
* UI and overlays remain at output resolution. Display color management remains in
  Blender's existing viewport pipeline.
* The feature handle, textures and history belong to a Film instance. Different
  viewports never share a feature handle or parameter map. SDK initialization is shared.
* Size/quality changes and destruction wait for pending GPU work before releasing NGX
  resources. Normal frames do not introduce a CPU/GPU completion wait.
* A viewport NGX failure produces a temporary defined image and selects native EEVEE
  on the next update. Final rendering discards the failed reconstruction and restarts
  once at native resolution. Temporary scaled images are never accepted as final output.

Final render behavior
---------------------

Each output frame, view and View Layer creates independent history. The scene time stays
fixed while SR/RR evaluates once per sample. Static-frame motion is zero; subpixel jitter
is passed separately. Render samples are raised to at least 32, shown in render progress.
The final reconstructed sample bypasses Film TAA. A compute adapter restores bottom-left
orientation and opaque alpha before Film Combined texture lookup/readback, Render Result,
ordinary Combined compositing, PNG or linear EXR output. No history is borrowed from
another animation frame, including when rendering frames in reverse order.

Temporal behavior
-----------------

Jitter uses a persistent Halton sequence in input-pixel units, independent of EEVEE's
navigation sample resets. For top-left NGX inputs, projection translation becomes
``(jitter.x, -jitter.y)``. History is reset on activation, size/quality changes, camera
identity/projection changes, backward or skipped timeline frames, large camera changes,
and depsgraph edits at a fixed timeline time. Ordinary navigation and consecutive
animation frames use motion reprojection. History decisions use the current synced
camera and are committed only for rendered samples, not display-only redraws.

Validation
----------

Run ``VKRenderGraphTestCompute.external_compute_dependencies_and_rebind`` in
``blender_test``. This checks producer/consumer synchronization and re-binding of the
same pipeline and descriptor set after external compute.

The viewport test must run in a windowed process, not ``--background``::

  blender --factory-startup --gpu-backend vulkan --debug-gpu --python tests/python/eevee_dlss_viewport.py -- --output captures

It saves composited PNG captures and a JSON report for native rendering, all SR modes, navigation,
settling, camera cuts, object motion, overlays, fallback, re-enabling, viewport splitting
and disabling. Also run it with ``--gpu-backend opengl`` for backend fallback. Inspect
the console for ``DLSS SR`` lines with successful NGX status and both retained/reset
history, and for Vulkan validation errors. A finite/non-empty capture alone is not proof
that DLSS executed or that temporal image quality is acceptable.

On Windows, enable the installed Khronos validation layer and synchronization validation
in the process environment before starting the test::

  $env:VK_INSTANCE_LAYERS = 'VK_LAYER_KHRONOS_validation'
  $env:VK_LAYER_VALIDATE_SYNC = '1'

Add ``--hardware-raytracing`` after the script's ``--output`` argument to test SR together
with this branch's EEVEE hardware ray tracing. The test also exits with live SR handles
in two viewports to exercise GPU completion and SDK teardown.

Inspect thin geometry, checkerboard detail, disocclusion, moving silhouettes and overlays
against native rendering. Production acceptance also needs representative content with
hair, transparency and animated materials; those have more demanding motion/coverage
semantics. This first version does not claim full-resolution overlay-depth accuracy.

Original SR baseline validation (2026-10-11, before RR/final-render extension)
---------------------------------------------------------------------------

Release build, SDK 310.4.0, RTX 4070 SUPER, NVIDIA driver 610.88:

* Vulkan render-graph tests: 51 passed, one existing scheduler case skipped.
* Vulkan SR: 13 captures, 1,141 successful NGX evaluations, retained and reset history.
* Vulkan SR with hardware RT: 13 captures, 1,109 successful NGX evaluations, two
  independent active viewports, disable/re-enable, and clean exit with live features.
* Both Vulkan runs had zero validation errors with synchronization validation enabled.
  Existing unused fragment-output warnings also appear before SR is enabled.
* OpenGL fallback: 13 captures, no NGX evaluations, clean exit.
* A 64x64 final render with Quality selected completed on native EEVEE without NGX.

Visual inspection caught and corrected an inverted jitter sign. Static checkerboard
edges and thin bars now resolve without that artifact. Stochastic lighting noise remains:
SR does not substitute for Ray Reconstruction or high-sample offline rendering. These
checks establish the integration baseline, not production acceptance across all content.

References
----------

* https://github.com/NVIDIA/DLSS/tree/v310.4.0
* https://github.com/NVIDIA/DLSS/blob/v310.4.0/doc/DLSS_Programming_Guide_Release.pdf
* https://github.com/NVIDIA/DLSS/blob/v310.4.0/doc/DLSS-RR%20Integration%20Guide.pdf

RR/final-render diagnostics
---------------------------

Add ``--rr`` to the viewport runner for the supported opaque RR subset. The final-render
runner writes PNG, EXR, raw arrays and timing records::

  blender -b --factory-startup --gpu-backend vulkan --debug-gpu --python tests/python/eevee_dlss_render.py -- --output captures --mode RR --sequence 1,2,1

Repeat with ``--mode SR``, ``--mode NATIVE --samples 256``, reverse frame order and each
``--fallback`` case. Repeat without ``-b`` for windowed final output. Compare the GPU logs,
not just images: successful ``DLSS RR`` and ``DLSS RR tracing`` lines confirm actual RR
evaluation and native ray-denoiser bypass. ``external_rr_inputs_and_reuse`` tests the seven
input dependencies and subsequent guide reuse in the Vulkan render graph.

``BLENDER_DLSS_DUMP_DIR`` opts into first-sample guide readback (width/height/channels in
``.txt``, float32 data in ``.f32``). Production has no guide readback. With ``--debug-gpu``,
``BLENDER_DLSS_TEST_FAIL_AT=3`` injects an NGX failure at evaluation index 3 to exercise the
native retry. Unset these variables for performance measurement.

``BLENDER_DLSS_PROFILE=1`` records Vulkan timestamps for the first 64 NGX evaluations
and reports SDK allocated VRAM at feature destruction. This measures the NGX phase,
not total EEVEE GPU time. Query pools are absent when profiling is disabled.

Validate dumped guides with ``tests/python/eevee_dlss_guides.py``. Run the asynchronous
F12 job and cancellation/recovery harness in a windowed process::

  blender --factory-startup --gpu-backend vulkan --debug-gpu --python tests/python/eevee_dlss_cancel.py -- --output captures --complete

Add ``--sr`` for SR. Omit ``--complete`` to inject cancellation after three samples,
then render another frame to check recovery. ``BLENDER_DLSS_TEST_CANCEL_AT`` is a
debug-GPU-only hook through the normal render cancellation gate; this test does not
simulate a physical Escape key. Use the same saved ``.blend`` for cross-process
determinism comparisons (``--save-fixture`` and ``--use-existing-scene`` in the render
runner), rather than separately generating a new fixture in each process.

RR and final-render validation record (2026-10-11)
------------------------------------------------

Release build and installation succeeded with SDK 310.4.0 on RTX 4070 SUPER,
driver 610.88. The final implementation retains experimental status.

* Render-graph suite: 52 passed, one pre-existing scheduler test skipped.
* Final viewport regressions: 13 SR captures and 15 RR captures, including all quality
  levels, motion, camera changes, live RR/SR switching, independent split viewports,
  fallback, reactivation and destruction with active features.
* Actual NGX SR/RR success and RR native-denoiser bypass were checked in logs.
  Selected Vulkan runs with synchronization validation had no validation errors.
* Final SR and RR completed through background rendering and the asynchronous F12
  render job. Quality, Balanced and Performance produced the requested output size.
  PNG, linear EXR and Combined-only exposure compositing were checked.
* Identical saved-scene RR frames rendered in forward order, reverse order and a
  separate single-frame process were bit-identical. Windowed versus background
  output was also bit-identical. These results are specific to this fixed environment.
* Mid-render NGX fault injection discarded reconstruction and retried natively once;
  its output was bit-identical to the matching native render. Cancellation followed
  by a fresh render completed without stale results or validation errors.
* Twelve native fallback cases passed: transparent film, depth pass, DOF, border,
  AOV, Cryptomatte, motion blur, transmission, normal-mapped unsupported hardware-RT
  material, volume, point cloud and multiview. Missing RR runtime retained SR;
  missing SR runtime and OpenGL used native rendering.
* Guide readback checked finite values, unit world normals, roughness range,
  reflectance, hit-distance/background conventions and zero final-frame motion.

The opaque PBR fixture includes diffuse, metallic, mirror and rough reflections,
thin bars, moving geometry and camera changes. Captures were visually inspected for
orientation, gross ghosting, reflection stability and noise. Against a native
256-sample reference, linear RGB mean absolute error was 0.0104024 for SR and
0.00689949 for RR. This scalar metric does not establish general perceptual quality.

At 1920x1080 and 32 samples, this simple fixture measured:

====================== =========== =========== ===========
Measurement            Native      SR          RR
====================== =========== =========== ===========
Render wall time       0.548 s     1.244 s     1.274 s
NGX steady GPU mean    n/a         0.653 ms    2.373 ms
Sampled GPU peak       4072 MiB    3990 MiB    4025 MiB
SDK allocated VRAM     n/a         94.0 MB     160.1 MB
====================== =========== =========== ===========

Wall time includes initialization/readback/output overhead. GPU means exclude the
first evaluation. Memory peaks are sampled whole-device ``nvidia-smi`` values with
approximately 2395 MiB baseline; they are not exact per-process peaks. Native was
faster on this fixture. Total EEVEE GPU timing and exact allocation peaks remain
unmeasured, and no speedup is claimed.

Local evidence is in ``E:/gitrepo/dlss_rr_validation/validation_summary.json``, with
images, raw arrays, guide dumps and per-case logs beside it. Older failed harness
iterations remain in that directory; the summary explicitly selects final runs.

Outstanding acceptance includes physical non-RTX hardware, broad production scenes,
long animation/disocclusion sequences and exhaustive perceptual review. Normal-mapped
and layered secondary-hit materials currently select native rendering; successful
fallback is not acceptance of those materials under RR. Other unsupported effects and
data workflows likewise retain the native path. These limits prevent claiming full
production visual acceptance from API success alone.
