#pragma once
#include <limits>
#include <memory>
#include <vector>

#include "marco.h"

#include "hittable.h"

class hittable_list : public hittable {
public:
    hittable_list() = default;
    explicit hittable_list(std::shared_ptr<hittable> object) { add(object); }

    void clear() { objects.clear(); }
    void add(std::shared_ptr<hittable> object) { objects.push_back(object); }

    virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const;

public:
    std::vector<std::shared_ptr<hittable>> objects;
};

bool hittable_list::hit(const ray& r, double t_min, double t_max, hit_record& rec) const
{
    hit_record temp_rec;
    bool       hit_anything   = false;
    auto       closest_so_far = std::numeric_limits<double>::max();

    for (const auto& object : objects)
    {
        if (object->hit(r, t_min, closest_so_far, temp_rec))
        {
            hit_anything   = true;
            rec            = temp_rec;
            closest_so_far = temp_rec.t;
        }
    }

    return hit_anything;
}
