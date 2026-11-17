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
}
