use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;

const DATA_FILE: &'static str = "./data/spatial_data.csv";
const META_FILE: &'static str = "./data/spatial_metadata.csv";
const GENE_CELL_CUTOFF: usize = 3000;

fn main() {
    // load the data
    let cell_data = data_loader_spatial();
    println!("Number of cells: {}", cell_data.len());
    // find the similar cells (exact same if any)
    let similarities = find_similar_cells(&cell_data);
    let mut count_0_distance = 0;
    for distance in similarities.iter().take(10_000) {
        if distance < &0.0001 {
            println!("Cells MSE distance = {:.4}", distance);
            count_0_distance += 1;
        }
    }
    println!("{}", count_0_distance);
    // make the initial graph using pet graph

    // connect the exact similar cells

    // add neighbouring cells to graph

    // cluster the graph
}

fn mean_squared_error(a: &[u16], b: &[u16]) -> f64 {
    let sum: f64 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = *x as f64 - *y as f64;
            diff * diff
        })
        .sum();
    sum / a.len() as f64
}

fn find_similar_cells(cells: &Vec<CellData>) -> Vec<f64> {
    let mut distances = Vec::new();
    for i in 0..cells.len() {
        for j in (i + 1)..cells.len() {
            let cell_a = &cells[i];
            let cell_b = &cells[j];
            if cell_a.read_counts.len() != cell_b.read_counts.len() {
                continue;
            }
            let mse = mean_squared_error(&cell_a.read_counts, &cell_b.read_counts);
            distances.push(mse);
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(&b).unwrap());
    distances
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
            if read_count > 1 {
                gene_expressed_by_cells += 1;
            }
        }
        if gene_expressed_by_cells > GENE_CELL_CUTOFF {
            println!("gene {} passed", line_index);
            for (cell_index, value) in values.iter().enumerate() {
                // convert to u32 and add to cell data
                let read_count = value.to_string().parse::<u16>().unwrap();
                all_cell_data[cell_index].read_counts.push(read_count);
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
    cell_type: String,
    spatial_location: (usize, usize)
}
impl CellData {
    fn new(cell_id: String, spatial_xy: (usize, usize)) -> CellData {
        CellData{
            cell_id: cell_id,
            read_counts: vec![],
            cell_type: String::new(),
            spatial_location: spatial_xy
        }
    }
}