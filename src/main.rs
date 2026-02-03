use std::fs::File;
use std::io::{BufRead, BufReader};
const DATA_FILE: &'static str = "./data/spatial_data.csv";
const META_FILE: &'static str = "./data/spatial_metadata.csv";

fn main() {
    // load the data
    data_loader_spatial();
    // make the initial graph

    // connect the exact similar cells

    // add neighbouring cells to graph

    // cluster the graph
}

fn data_loader_spatial() -> (Vec<CellData>, Vec<String>) {
    println!("Start Meta Data Loading");
    let meta_file = File::open(META_FILE).expect("cannot open data file");
    let meta_data_reader = BufReader::new(meta_file);
    let mut meta_data_info = vec![];
    for (line_index, line) in meta_data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        if line_index != 0 {
            meta_data_info.push((values[0].trim().to_string(), values[35].to_string(), values[36].to_string()));
        }
    }
    println!("Start Data Loading");
    let data_file = File::open(DATA_FILE).expect("cannot open data file");
    let data_reader = BufReader::new(data_file);
    
    let mut all_cell_data= vec![];
    let mut cell_ids = vec![];
    for (line_index, line) in data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        if line_index == 0 {
            // this is the header, make new cell vector
            all_cell_data = vec![CellData::new(0); values.len()];
            // save the values in a vector, cell id_fov_etc
            for value in values {
                cell_ids.push(value.trim().to_string());
            }
            continue;
        }
        let mut gene_expressed_by_cells = 0;
        for (_cell_index, value) in values.iter().enumerate() {
            // convert to u32 and add to cell data
            let read_count = value.to_string().parse::<u16>().unwrap();
            if read_count > 1 {
                gene_expressed_by_cells += 1;
            }
        }
        if gene_expressed_by_cells > 3000 {
            println!("gene {} passed", line_index);
            for (cell_index, value) in values.iter().enumerate() {
                // convert to u32 and add to cell data
                let read_count = value.to_string().parse::<u16>().unwrap();
                all_cell_data[cell_index].read_counts.push(read_count);
                all_cell_data[cell_index].gene_count = all_cell_data[cell_index].gene_count + 1;
            }
        }
    }
    println!("End Data Loading");
    (all_cell_data, cell_ids)
}

#[derive(Clone)]
struct CellData {
    cell_id: String,
    read_counts: Vec<u16>,
    gene_count: usize,
    cell_type: String,
    spatial_location: (usize, usize)
}
impl CellData {
    fn new(gene_count: usize) -> CellData {
        CellData{
            cell_id: String::new(),
            read_counts: vec![0; gene_count],
            gene_count: gene_count,
            cell_type: String::new(),
            spatial_location: (0, 0)
        }
    }
}