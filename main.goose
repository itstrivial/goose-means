import std;
import checks;

const cluster_count = 5;
const epochs = 10;
const batch_size = 50;
const alpha = 0.25;
const run_count = 20;

struct Point {
    x: f64,
    y: f64,
}

struct Cloud {
    center: Point,
    count:  i64,
}

struct ClusterStats {
    sum:   Point,
    count: i64,
}

struct EmaStats {
    sum:   Point,
    count: f64,
}

struct EpochLoss {
    lloyd: f64,
    ema:   f64,
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
    each_chunk(points, batch_size) { batch, _ =>
        ema_update(batch, centroids, running, alpha);
    };
}

fn generate_dataset() -> Point[>..] {
    const clouds: Cloud[cluster_count] = [
        Cloud { center: Point { -6.0, -4.0 }, count: 400 },
        Cloud { center: Point { 6.0, -4.0 }, count: 250 },
        Cloud { center: Point { 0.0, 0.0 }, count: 170 },
        Cloud { center: Point { -6.0, 4.0 }, count: 110 },
        Cloud { center: Point { 6.0, 4.0 }, count: 70 },
    ];
    const sigma = 2.5;
    const seed: u64 = 42;

    var generator = rng { seed };
    var points: Point[>..] = [];
    for cloud in clouds {
        for sample in cloud.count {
            let dx = sigma * generator.rand_gaussian();
            let dy = sigma * generator.rand_gaussian();
            points.push(Point { x: cloud.center.x + dx, y: cloud.center.y + dy });
        }
    }

    shuffle(points, generator);
    return points;
}

fn compare(points: const Point[:], initial_centroids: Point[cluster_count], shuffle_seed: u64) -> EpochLoss[epochs + 1] {
    var shuffled_points: Point[>..] = [];
    shuffled_points.append(points);
    var shuffle_rng = rng { shuffle_seed };

    var lloyd_centroids = initial_centroids;
    var ema_centroids = initial_centroids;
    var running: EmaStats[cluster_count] = [EmaStats { .. }; cluster_count];
    var losses:  EpochLoss[epochs + 1]   = [EpochLoss { .. }; epochs + 1];

    for epoch in epochs + 1 {
        if epoch > 0 {
            lloyd_update(points, lloyd_centroids);
            shuffle(shuffled_points, shuffle_rng);
            ema_epoch(shuffled_points, ema_centroids, running, batch_size, alpha);
        }
        losses[epoch] = EpochLoss { lloyd: mse(points, lloyd_centroids), ema: mse(points, ema_centroids) };
    }
    return losses;
}

fn sample_centroids(points: const Point[:], seed: u64) -> Point[cluster_count] {
    assert(points.len >= cluster_count);
    var candidates: Point[>..] = [];
    candidates.append(points);
    var sample_rng = rng { seed };
    shuffle(candidates, sample_rng);

    var centroids: Point[cluster_count] = [Point { .. }; cluster_count];
    copy_into(centroids[..], candidates[..cluster_count]);
    return centroids;
}

fn median(sorted_values: const f64[:]) -> f64 {
    assert(sorted_values.len > 0);
    let midpoint = sorted_values.len / 2;
    if sorted_values.len % 2 == 0 {
        return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2.0;
    }
    return sorted_values[midpoint];
}

fn append_mse(row: u8[>..]&, error: f64, width: i64) {
    var cell: u8[>..] = [];
    format_flt(cell, error, 4);
    for i in max(0, width - cell.len) {
        row.push(' ');
    }
    row.append(cell);
}

fn run_experiment() {
    const points = generate_dataset();
    const tie_tolerance = 1e-9;
    var curves: EpochLoss[epochs + 1][run_count] = [[EpochLoss { .. }; epochs + 1]; run_count];
    var ema_wins = 0;

    print("Lloyd vs. EMA method.");
    print(points.len, " points in ", cluster_count, " overlapping clouds.");
    print(run_count, " runs. Both methods start at the same centroids in each run.");
    print(epochs, " epochs, ", batch_size, " points per mini-batch, step size ", alpha, ".");

    for run in run_count {
        let seed = (run + 1) as u64;
        const initial_centroids = sample_centroids(points, seed);
        curves[run] = compare(points, initial_centroids, 1000 + seed);
        let final = curves[run][epochs];
        if final.ema - final.lloyd < -tie_tolerance {
            ema_wins += 1;
        }
    }

    print("");
    print("Mean squared error (median across runs):");
    print("Epoch |   Lloyd | EMA");
    for epoch in epochs + 1 {
        var lloyd_errors: f64[run_count] = [0.0; run_count];
        var ema_errors:   f64[run_count] = [0.0; run_count];
        for run in run_count {
            let loss = curves[run][epoch];
            lloyd_errors[run] = loss.lloyd;
            ema_errors[run] = loss.ema;
        }
        sort(lloyd_errors);
        sort(ema_errors);

        var row: u8[>..] = [];
        format_int(row, epoch, 10, 5, ' ');
        row.append(" | ");
        append_mse(row, median(lloyd_errors), 7);
        row.append(" | ");
        append_mse(row, median(ema_errors), 10);
        print(row);
    }
    print("");
    print("After ", epochs, " epochs, EMA method had lower error in ", ema_wins, " of ", run_count, " runs.");
}

fn main() {
    check_full_batch_equivalence();
    check_ema_epoch();
    print("All checks passed.");
    print("");
    run_experiment();
}
