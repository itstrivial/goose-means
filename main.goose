import std;

struct Point {
    x: f64,
    y: f64,
}

fn squared_distance(first: Point, second: Point) -> f64 {
    let dx = first.x - second.x;
    let dy = first.y - second.y;
    return dx * dx + dy * dy;
}

fn mse(points: const Point[:], centroids: const Point[:]) -> f64 {
    assert(points.len > 0);
    assert(centroids.len > 0);

    var total_squared_distance: f64 = 0.0;
    for point in points {
        var nearest_squared_distance = squared_distance(point, centroids[0]);
        for centroid_index in 1..centroids.len {
            let candidate_squared_distance = squared_distance(point, centroids[centroid_index]);
            if candidate_squared_distance < nearest_squared_distance {
                nearest_squared_distance = candidate_squared_distance;
            }
        }
        total_squared_distance += nearest_squared_distance;
    }

    return total_squared_distance / points.len;
}

fn generate_dataset(centers: const Point[:]) -> Point[>..] {
    const points_per_center = 200;
    const sigma = 1.5;
    const seed: u64 = 42;

    var generator = rng { seed };
    var points: Point[>..] = [];
    for center in centers {
        for sample in points_per_center {
            let dx = sigma * generator.rand_gaussian();
            let dy = sigma * generator.rand_gaussian();
            points.push(Point { x: center.x + dx, y: center.y + dy });
        }
    }

    shuffle(points, generator);
    return points;
}

fn check_four_point_example() {
    const points: Point[4] = [
        Point { x: 0.0, y: 0.0 },
        Point { x: 0.0, y: 2.0 },
        Point { x: 8.0, y: 0.0 },
        Point { x: 8.0, y: 2.0 },
    ];
    const initial_centroids: Point[2] = [
        Point { x: 0.0, y: 0.0 },
        Point { x: 8.0, y: 0.0 },
    ];
    const manually_moved_centroids: Point[2] = [
        Point { x: 0.0, y: 1.0 },
        Point { x: 8.0, y: 1.0 },
    ];

    assert(squared_distance(points[1], initial_centroids[1]) == 68.0);

    let initial_mse = mse(points, initial_centroids);
    let moved_mse = mse(points, manually_moved_centroids);

    assert(initial_mse == 2.0);
    assert(moved_mse == 1.0);

    print("Four-point checks passed (mean squared errors ", initial_mse, " and ", moved_mse, ").");
}

fn main() {
    check_four_point_example();

    const generating_centers: Point[5] = [
        Point { x: -6.0, y: -4.0 },
        Point { x: 6.0, y: -4.0 },
        Point { x: 0.0, y: 0.0 },
        Point { x: -6.0, y: 4.0 },
        Point { x: 6.0, y: 4.0 },
    ];
    const initial_centroids: Point[5] = [
        Point { x: -4.0, y: -3.0 },
        Point { x: 4.0, y: -2.0 },
        Point { x: 2.0, y: 1.0 },
        Point { x: -5.0, y: 2.0 },
        Point { x: 5.0, y: 2.0 },
    ];

    const points = generate_dataset(generating_centers);
    print("Generated ", points.len, " points around ", generating_centers.len, " centers.");
    print("Mean squared error before training: ", mse(points, initial_centroids));
}
