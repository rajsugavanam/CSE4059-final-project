# REDUCTION TESTS ------------------------------
add_executable(reduction_test "reduction_test.cu")

target_link_libraries(reduction_test
    reduction
)

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
target_link_libraries(obj_reduction_test
    reduction
)
add_test(NAME Obj.Reduce.AABB
    COMMAND obj_reduction_test
)