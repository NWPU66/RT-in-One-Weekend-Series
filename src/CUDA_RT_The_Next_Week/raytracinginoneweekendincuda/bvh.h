#ifndef BVHH
#define BVHH

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iostream>

#include <curand_kernel.h>
#include <curand_normal.h>
#include <curand_uniform.h>

#include "aabb.h"
#include "hitable.h"
#include "hitable_list.h"
#include "util.h"

class bvh_node : public hitable {
public:
    __device__ bvh_node() {}
    __device__ bvh_node(hitable_list& list, float time0, float time1, curandState* rand_state)
        : bvh_node(list.list, 0, list.list_size, time0, time1, rand_state)
    {
    }
    __device__
    bvh_node(hitable** list, int start, int end, float time0, float time1, curandState* rand_state);
    __device__ ~bvh_node()
    {
        delete left;
        delete right;
        left = right = object = nullptr;
    }

    __device__ virtual bool
    hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;
    __device__ virtual bool bounding_box(float t0, float t1, aabb& output_box) const override;

    __device__ float sdf(const vec3& p, float time) const override;

public:
    // object 指向场景中的其他图元，析构的时候不会释放，释放由free_world()负责
    hitable *left, *right, *object;
    aabb     box;
};

template <int axis> __device__ inline bool box_compare(const hitable* a, const hitable* b)
{
    aabb box1, box2;
    if (!a->bounding_box(0, 0, box1) || !b->bounding_box(0, 0, box2))
    {
        // std::cerr << "No bounding box in bvh_node constructor.\n";
        // TODO - 该代码在GPU中，要向CPU发回错误信息
    }

    return box1.min().e[axis] <= box2.min().e[axis];
}

__host__ __device__ void
bubble_sort2(hitable** x, size_t start, size_t end, bool cmp(const hitable*, const hitable*))
{
    for (int i = start; i < end - 1; i++)
    {
        bool swapped = false;
        for (int j = start; j < end - i - 1; j++)
        {
            if (!cmp(x[j], x[j + 1]))
            {
                swapped       = true;
                hitable* temp = x[j];
                x[j]          = x[j + 1];
                x[j + 1]      = temp;
            }
        }

        if (!swapped) { break; }
    }
}

__device__ bvh_node::bvh_node(hitable**    list,
                              int          start,
                              int          end,
                              float        time0,
                              float        time1,
                              curandState* rand_state)
{
    int  axis       = int(curand_uniform(rand_state) * 2.999f);
    auto comparator = (axis == 0) ? box_compare<0> : (axis == 1) ? box_compare<1> : box_compare<2>;

    int object_span = end - start;
    switch (object_span)
    {
        case 1: {
            left = right = nullptr;
            object       = list[start];

            aabb temp_box;
            if (object->bounding_box(time0, time1, temp_box)) { box = temp_box; }
            else
            {
                // std::cerr << "No bounding box in bvh_node constructor.\n";
                // TODO - 该代码在GPU中，要向CPU发回错误信息
            }
            break;
        }
        case 2: {
            if (comparator(list[start], list[start + 1]))
            {
                left  = new bvh_node(list, start, start + 1, time0, time1, rand_state);
                right = new bvh_node(list, start + 1, end, time0, time1, rand_state);
            }
            else
            {
                right = new bvh_node(list, start, start + 1, time0, time1, rand_state);
                left  = new bvh_node(list, start + 1, end, time0, time1, rand_state);
            }
            object = nullptr;

            aabb box_left, box_right;
            if (!left->bounding_box(time0, time1, box_left) ||
                !right->bounding_box(time0, time1, box_right))
            {
                // std::cerr << "No bounding box in bvh_node constructor.\n";
                // TODO - 该代码在GPU中，要向CPU发回错误信息
            }
            box = surrounding_box(box_left, box_right);

            break;
        }
        default: {  // 先序遍历创建bvh tree
            bubble_sort2(list, start, end, comparator);

            // test sort result
            bool t1=comparator(list[start], list[start + 1]);
            bool t2=comparator(list[start + 1], list[start + 2]);
            bool t3=comparator(list[start + 2], list[start + 3]);

            auto mid = (start + end) / 2;

            left  = new bvh_node(list, start, mid, time0, time1, rand_state);
            right = new bvh_node(list, mid, end, time0, time1, rand_state);

            object = nullptr;

            aabb box_left, box_right;
            if (!left->bounding_box(time0, time1, box_left) ||
                !right->bounding_box(time0, time1, box_right))
            {
                // std::cerr << "No bounding box in bvh_node constructor.\n";
                // TODO - 该代码在GPU中，要向CPU发回错误信息
            }
            box = surrounding_box(box_left, box_right);

            //test bounding box 
            vec3 a = box.min();
            vec3 b = box.max();

            break;
        }
    }
}

__device__ bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const
{
    //test 
    vec3 a = box.min();
    vec3 b = box.max();

    if (!box.hit(r, t_min, t_max)) { return false; }

    if (object)
    {
        // 如果object不空，说明来到bvh的叶子节点，调用object的hit方法
        object->hit(r, t_min, t_max, rec);
    }
    else
    {
        // 非叶子节点，递归调用左右子节点的hit方法
        bool hit_left  = left->hit(r, t_min, t_max, rec);
        bool hit_right = right->hit(r, t_min, hit_left ? rec.t : t_max, rec);

        return hit_left || hit_right;
    }
}

__device__ bool bvh_node::bounding_box(float t0, float t1, aabb& output_box) const
{
    output_box = box;
    return true;
}

__device__ float bvh_node::sdf(const vec3& p, float time) const
{
    if (object) { return object->sdf(p, time); }
    float left_sdf  = left->sdf(p, time);
    float right_sdf = right->sdf(p, time);
    return (abs(left_sdf) < abs(right_sdf)) ? left_sdf : right_sdf;
}

#endif