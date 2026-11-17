import std;

struct Point {
    x: f64,
    y: f64,
}

struct ClusterStats {
    sum:   Point,
    count: i64,
}

struct EmaStats {
    sum:   Point,
    count: f64,
}

fn squared_distance(first: Point, second: Point) -> f64 {
    let dx = first.x - second.x;
    let dy = first.y - second.y;
    return dx * dx + dy * dy;
}

fn nearest_centroid(point: Point, centroids: const Point[:]) -> i64, f64 {
    var nearest_index = 0;
    var nearest_distance = squared_distance(point, centroids[0]);
    for index in 1..centroids.len {
        let distance = squared_distance(point, centroids[index]);
        if distance < nearest_distance {
            nearest_index = index;
            nearest_distance = distance;
        }
    }
    return nearest_index, nearest_distance;
}

fn mse(points: const Point[:], centroids: const Point[:]) -> f64 {
    assert(points.len > 0);
    assert(centroids.len > 0);

    var total_squared_distance: f64 = 0.0;
    for point in points {
        let _, distance = nearest_centroid(point, centroids);
        total_squared_distance += distance;
    }

    return total_squared_distance / points.len;
}

fn batch_stats(points: const Point[:], centroids: const Point[:]) -> ClusterStats[>..] {
    assert(centroids.len > 0);

    var stats: ClusterStats[>..] = [];
    stats.resize(centroids.len, ClusterStats { .. });
    for point in points {
        let index = nearest_centroid(point, centroids);
        stats[index].sum.x += point.x;
        stats[index].sum.y += point.y;
        stats[index].count += 1;
    }
    return stats;
}

fn lloyd_update(points: const Point[:], centroids: Point[:]) {
    const stats = batch_stats(points, centroids);
    for cluster, index in stats {
        if cluster.count > 0 {
            centroids[index] = Point {
                x: cluster.sum.x / cluster.count,
                y: cluster.sum.y / cluster.count,
            };
        }
    }
}

fn ema_update(batch: const Point[:], centroids: Point[:], running: EmaStats[:], alpha: f64) {
    assert(alpha > 0.0 && alpha <= 1.0);
    assert(running.len == centroids.len);

    const stats = batch_stats(batch, centroids);
    let decay = 1.0 - alpha;
    for cluster, index in stats {
        let previous = running[index];
        let sum = Point {
            x: decay * previous.sum.x + alpha * cluster.sum.x,
            y: decay * previous.sum.y + alpha * cluster.sum.y,
        };
        let count = decay * previous.count + alpha * cluster.count;
        running[index] = EmaStats { sum: sum, count: count };

        if cluster.count > 0 {
            centroids[index] = Point { x: sum.x / count, y: sum.y / count };
        }
    }
}

fn ema_epoch(points: const Point[:], centroids: Point[:], running: EmaStats[:], batch_size: i64, alpha: f64) {
    assert(batch_size > 0);

    var start = 0;
    while start < points.len {
        let end = start + min(batch_size, points.len - start);
        ema_update(points[start..end], centroids, running, alpha);
        start = end;
    }
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

fn check_full_batch_equivalence() {
    const points: Point[4] = [
        Point { x: 0.0, y: 0.0 },
        Point { x: 0.0, y: 2.0 },
        Point { x: 8.0, y: 0.0 },
        Point { x: 8.0, y: 2.0 },
    ];
    const initial_centroids: Point[3] = [
        Point { x: 0.0, y: 0.0 },
        Point { x: 8.0, y: 0.0 },
        Point { x: 40.0, y: 40.0 },
    ];
    const expected_centroids: Point[3] = [
        Point { x: 0.0, y: 1.0 },
        Point { x: 8.0, y: 1.0 },
        Point { x: 40.0, y: 40.0 },
    ];

    assert(squared_distance(points[1], initial_centroids[1]) == 68.0);
    let initial_mse = mse(points, initial_centroids);
    assert(initial_mse == 2.0);

    var lloyd_centroids = initial_centroids;
    lloyd_update(points, lloyd_centroids);
    assert(lloyd_centroids == expected_centroids);
    assert(mse(points, lloyd_centroids) == 1.0);

    var ema_centroids = initial_centroids;
    var running: EmaStats[3] = [EmaStats { .. }; 3];
    // Give every cluster history without moving its starting centroid.
    ema_update(initial_centroids, ema_centroids, running, 0.5);
    assert(ema_centroids == initial_centroids);
    assert(running[2].count == 0.5);

    ema_update(points, ema_centroids, running, 1.0);
    assert(ema_centroids == lloyd_centroids);
    assert(running[2] == EmaStats { .. });

    print("Four-point Lloyd and full-batch checks passed.");
}

fn check_ema_epoch() {
    const points: Point[6] = [
        Point { x: 0.0, y: 0.0 },
        Point { x: 0.0, y: 1.0 },
        Point { x: 0.0, y: 1.0 },
        Point { x: 0.0, y: 2.0 },
        Point { x: 10.0, y: 2.0 },
        Point { x: 0.0, y: 4.0 },
    ];
    var centroids: Point[2] = [
        Point { x: -1.0, y: 1.0 },
        Point { x: 11.0, y: 1.0 },
    ];
    var running: EmaStats[2] = [EmaStats { .. }; 2];

    ema_epoch(points, centroids, running, 5, 0.5);

    assert(running[0] == EmaStats { sum: Point { x: 0.0, y: 3.0 }, count: 1.5 });
    assert(centroids[0] == Point { x: 0.0, y: 2.0 });
    assert(running[1] == EmaStats { sum: Point { x: 2.5, y: 0.5 }, count: 0.25 });
    assert(centroids[1] == Point { x: 10.0, y: 2.0 });

    print("Exponential moving average checks passed.");
}

fn main() {
    check_full_batch_equivalence();
    check_ema_epoch();

    const epochs = 10;
    const batch_size = 50;
    const alpha = 0.25;
    const shuffle_seed: u64 = 43;

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
    var shuffled_points: Point[>..] = [];
    shuffled_points.append(points);
    var shuffle_rng = rng { shuffle_seed };

    var lloyd_centroids = initial_centroids;
    var ema_centroids = initial_centroids;
    var running: EmaStats[>..] = [];
    running.resize(ema_centroids.len, EmaStats { .. });

    print("Generated ", points.len, " points around ", generating_centers.len, " centers.");
    print("Mini-batch size: ", batch_size, ", step size: ", alpha);
    print("Full-dataset mean squared error");
    print("Epoch | Lloyd | Exponential moving average");
    for epoch in epochs + 1 {
        if epoch > 0 {
            lloyd_update(points, lloyd_centroids);
            shuffle(shuffled_points, shuffle_rng);
            ema_epoch(shuffled_points, ema_centroids, running, batch_size, alpha);
        }
        print(epoch, " | ", mse(points, lloyd_centroids), " | ", mse(points, ema_centroids));
    }
}
