use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;

const DATA_FILE: &'static str = "./data/spatial_data.csv";
const META_FILE: &'static str = "./data/spatial_metadata.csv";
const GENE_CELL_CUTOFF: usize = 3000;
const COUNT_DIFF_FOR_SIMILAR: usize = 100;
const CELL_CONNECT_DISTANCE: usize = 1000; //use L2 and see // for 1000 median 14 // for 500 median 5

fn main() {
    // load the data
    let cell_data = data_loader_spatial();
    println!("Number of cells: {}", cell_data.len());
    // find the similar cells (exact same if any)
    let (similarities, map) = find_similar_close_by_cells(&cell_data);
    let mut count_0_distance = 0;
    for distance in similarities.iter() {
        if distance.2 < 200.0 {
            println!("Cell A id {} loc {:?} total {} Cell B id {} loc {:?} total {} Gene_distance {}", distance.0.cell_id, distance.0.spatial_location, distance.0.total_count, distance.1.cell_id, distance.1.spatial_location, distance.1.total_count, distance.2);
             count_0_distance += 1;
        }
    }
    println!("Distances count: {}", count_0_distance);
    // check median of map
    let mut lengths: Vec<usize> = map.values().map(|v| v.len()).collect();
    let sum: usize = lengths.iter().sum();
    println!("Average: {}", sum as f64 / lengths.len() as f64);
    lengths.sort_unstable();
    println!("Median: {}", lengths[lengths.len() / 2] as f64);

    // make the initial graph using pet graph

    // connect the exact similar cells

    // add neighbouring cells to graph

    // cluster the graph
}

fn squared_error(a: &[u16], b: &[u16]) -> f64 {
    let sum: f64 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = *x as f64 - *y as f64;
            diff * diff
        })
        .sum();
    sum as f64
}

fn find_similar_close_by_cells(cells: &Vec<CellData>) -> (Vec<(&CellData, &CellData, f64)>, HashMap<(usize, usize), Vec<&CellData>>)  {
    // squared distance of read count difference in two cells
    let mut gene_distances = Vec::new();
    // hash map for keeping track of close by cells for each coordinate
    let mut map: HashMap<(usize, usize), Vec<&CellData>> = HashMap::new();
    for i in 0..cells.len() {
        let cell_a = &cells[i];
        let (cell_a_x, cell_a_y) = cell_a.spatial_location;
        // add to hashmap cell a
        map.insert((cell_a_x, cell_a_y), vec![cell_a]);
        for j in (i + 1)..cells.len() {
            let cell_b = &cells[j];
            let (cell_b_x, cell_b_y) = cell_b.spatial_location;
            // check read count vecs font match
            if cell_a.read_counts.len() != cell_b.read_counts.len(){
                continue;
            }
            // check if within count difference
            if cell_a.total_count.abs_diff(cell_b.total_count) > COUNT_DIFF_FOR_SIMILAR {
                continue;
            }
            // check if cells are within distance L2
            if (cell_a_x.abs_diff(cell_b_x).pow(2) + (cell_a_y.abs_diff(cell_b_y)).pow(2)).isqrt()
            > CELL_CONNECT_DISTANCE {
                continue;
            }
            // cell b is near spatial coordianates of cell a, add to hashmap
            if let Some(vec_value) = map.get_mut(&(cell_a_x, cell_a_y)) {
                vec_value.push(cell_b);
            }
            let mse = squared_error(&cell_a.read_counts, &cell_b.read_counts);
            gene_distances.push((cell_a, cell_b, mse));
        }
    }
    gene_distances.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    (gene_distances, map)
}

// load spatial data and populate celldata vec
fn data_loader_spatial() -> Vec<CellData> {
    println!("Start Meta Data Loading");
    // make a hashmap for look up of spatial locaiton of cell id
    let mut spatial_lookup: HashMap<String, (usize, usize)> = HashMap::new();
    let meta_file = File::open(META_FILE).expect("cannot open data file");
    let meta_data_reader = BufReader::new(meta_file);
    for (line_index, line) in meta_data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        if line_index != 0 {
            let x = values[35].parse().unwrap();
            let y = values[36].parse().unwrap();
            spatial_lookup.insert(values[0].trim().to_string(), (x, y));
        }
    }

    println!("Start Data Loading");
    let data_file = File::open(DATA_FILE).expect("cannot open data file");
    let data_reader = BufReader::new(data_file);
    let mut all_cell_data = vec![];
    for (line_index, line) in data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        // first line has the cell ids
        if line_index == 0 {
            // save the values in a vector, cell id_fov_etc
            for value in values {
                let cell_id = value.trim().to_string();
                // using the hashmap find the spatial location
                let (spatial_location_x, spatial_location_y) = spatial_lookup.get(&cell_id).cloned().unwrap_or((0, 0));
                let temp_cell_data = CellData::new(cell_id, (spatial_location_x, spatial_location_y));
                all_cell_data.push(temp_cell_data);
            }
            continue;
        }
        let mut gene_expressed_by_cells = 0;
        // only use the genes which are expressed by num of cells greater than GENE_CELL_CUTOFF
        for (_cell_index, value) in values.iter().enumerate() {
            // convert to u32 and add to cell data
            let read_count = value.to_string().parse::<u16>().unwrap();
            if read_count > 0 {
                gene_expressed_by_cells += 1;
            }
        }
        if gene_expressed_by_cells > GENE_CELL_CUTOFF {
            println!("gene {} passed", line_index);
            for (cell_index, value) in values.iter().enumerate() {
                // convert to u32 and add to cell data
                let read_count = value.to_string().parse::<u16>().unwrap();
                all_cell_data[cell_index].read_counts.push(read_count);
                all_cell_data[cell_index].total_count += read_count as usize;
                if read_count > 0 {
                    all_cell_data[cell_index].genes_with_count += 1;
                }
            }
        }
    }
    println!("End Data Loading");
    all_cell_data
}

#[derive(Clone)]
struct CellData {
    cell_id: String,
    read_counts: Vec<u16>,
    spatial_location: (usize, usize),
    total_count: usize,
    genes_with_count: usize,
}
impl CellData {
    fn new(cell_id: String, spatial_xy: (usize, usize)) -> CellData {
        CellData{
            cell_id: cell_id,
            read_counts: vec![],
            spatial_location: spatial_xy,
            total_count: 0,
            genes_with_count: 0
        }
    }
}