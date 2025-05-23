# TRIANGLE3 CRT TEST ------------------------------
add_executable(triangle3_crt_test "triangle3_crt_test.cu")
add_test(NAME Triangle3.CRT
    COMMAND triangle3_crt_test
)

# REDUCTION TESTS ------------------------------
add_executable(reduction_test "reduction_test.cu")

# target_link_libraries(reduction_test
#     reduction_old
# )

add_test(NAME Reduce.MinMax.1000
    COMMAND reduction_test 1000
)

add_test(NAME Reduce.MinMax.10000
    COMMAND reduction_test 100000
)

add_test(NAME Reduce.MinMax.100000
    COMMAND reduction_test 1000000
)

# AABB RENDER TEST ------------------------------
add_executable(aabb_render_test "aabb_render_test.cu")
add_test(NAME AABB.Render.Cube
    COMMAND aabb_render_test
)

# OBJ REDUCTION + AABB TEST ------------------------------
add_executable(obj_reduction_test "obj_reduction_test.cu")
# target_link_libraries(obj_reduction_test
#     reduction_old
# )
add_test(NAME Obj.Reduce.AABB
    COMMAND obj_reduction_test
)

# CAMERA TEST ------------------------------
add_executable(camera_test "camera_test.cu")
add_test(NAME Camera.Test
    COMMAND camera_test
)

# AABB RAII TEST ------------------------------
add_executable(aabb_raii_test "aabb_raii_test.cu")
target_link_libraries(aabb_raii_test
    aabb
)
add_test(NAME AABB.RAII.Cube
    COMMAND aabb_raii_test
)

# SCENE MANAGER TEST ------------------------------
add_executable(scene_manager_test "scene_manager_test.cu")
target_link_libraries(scene_manager_test
    aabb
    scene_manager
    reduction
)
target_include_directories(scene_manager_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include/util
)
add_test(NAME Scene.Manager.RAII
    COMMAND scene_manager_test
)

# TRIANGLE MESH TEST ------------------------------
add_executable(mesh_reduction_test "mesh_reduction_test.cu")
target_link_libraries(mesh_reduction_test
    aabb
    reduction
    triangle_mesh
    scene_manager
)
target_include_directories(mesh_reduction_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include/util
)
add_test(NAME Mesh.Reduce.AABB
    COMMAND mesh_reduction_test
)

# REDUCTION CONTINUED TEST ------------------------------
add_executable(reduction_benchmark "reduction_benchmark.cu")
target_link_libraries(reduction_benchmark
    reduction
)
add_test(NAME Reduce.Benchmark
    COMMAND reduction_benchmark
)
# LIGHT SAMPLING TEST ------------------------------
# add_executable(light_samp_test "light_sampling_test.cu")
# target_link_libraries(light_samp_test
#     light_sampling
# )
# add_test(NAME LightSamp.Test
#     COMMAND light_samp_test
# )

# Cornell Box Test ------------------------------
add_executable(cornell_box_test "cornell_box_test.cu")
target_link_libraries(cornell_box_test
    aabb
    reduction
    triangle_mesh
    scene_manager
)
add_test(NAME CornellBox.Test
    COMMAND cornell_box_test
)

# TRIANGLE MESH VEC TEST ------------------------------
add_executable(mesh_vec_test "mesh_vec_test.cu")
target_link_libraries(mesh_vec_test
    aabb
    reduction
    triangle_mesh
    scene_manager
)
add_test(NAME TriangleMesh.Vec
    COMMAND mesh_vec_test
)

# CORNELL BOX SPECTRAL TEST ------------------------------
add_executable(cornell_box_spectral_test "cornell_box_spectral_test.cu")
target_link_libraries(cornell_box_spectral_test
    aabb
    reduction
    triangle_mesh
    scene_manager
)
target_include_directories(cornell_box_spectral_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include/util
)
add_test(NAME CornellBox.Spectral
    COMMAND cornell_box_spectral_test
)
