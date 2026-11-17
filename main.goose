struct Point {
    horizontal: f64,
    vertical:   f64,
}

fn squared_euclidean_distance(first: Point, second: Point) -> f64 {
    let horizontal_difference = first.horizontal - second.horizontal;
    let vertical_difference = first.vertical - second.vertical;
    return horizontal_difference * horizontal_difference
         + vertical_difference * vertical_difference;
}

fn mean_squared_error(points: const Point[:], centroids: const Point[:]) -> f64 {
    assert(points.len > 0);
    assert(centroids.len > 0);

    var total_squared_distance: f64 = 0.0;
    for point in points {
        var nearest_squared_distance = squared_euclidean_distance(point, centroids[0]);
        for centroid_index in 1..centroids.len {
            let candidate_squared_distance = squared_euclidean_distance(point, centroids[centroid_index]);
            if candidate_squared_distance < nearest_squared_distance {
                nearest_squared_distance = candidate_squared_distance;
            }
        }
        total_squared_distance += nearest_squared_distance;
    }

    return total_squared_distance / points.len;
}

fn main() {
    let points: Point[4] = [
        Point { horizontal: 0.0, vertical: 0.0 },
        Point { horizontal: 0.0, vertical: 2.0 },
        Point { horizontal: 8.0, vertical: 0.0 },
        Point { horizontal: 8.0, vertical: 2.0 },
    ];
    let initial_centroids: Point[2] = [
        Point { horizontal: 0.0, vertical: 0.0 },
        Point { horizontal: 8.0, vertical: 0.0 },
    ];
    // Supplied by hand.
    let manually_moved_centroids: Point[2] = [
        Point { horizontal: 0.0, vertical: 1.0 },
        Point { horizontal: 8.0, vertical: 1.0 },
    ];

    // Check both coordinate differences.
    assert(squared_euclidean_distance(points[1], initial_centroids[1]) == 68.0);

    let initial_mean_squared_error = mean_squared_error(points, initial_centroids);
    let moved_mean_squared_error = mean_squared_error(points, manually_moved_centroids);

    assert(initial_mean_squared_error == 2.0);
    assert(moved_mean_squared_error == 1.0);

    print("Initial mean squared error: ", initial_mean_squared_error);
    print("Mean squared error with manually moved centroids: ", moved_mean_squared_error);
    print("All checks passed.");
}
