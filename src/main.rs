mod leiden_alg;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use petgraph::graphmap::UnGraphMap;
use petgraph::dot::{Config, Dot};
use leiden_alg::{Graph, TrivialModularityOptimizer};


const COUNT_FILE: &'static str = "./data/cerebellum_count.csv";
const COOR_FILE: &'static str = "./data/cerebellum_coor.csv";
const GLM_FILE: &'static str = "./data/cerebellum_glm.csv";
const GENE_CELL_CUTOFF: usize = 3000;
const COUNT_DIFF_FOR_SIMILAR: usize = 500;
//const CELL_CONNECT_DISTANCE: isize = 10_000; //use L2 and see // for 1000 median 14 // for 500 median 5

fn main() {
    // test leiden 
    //leiden_test();
    // load the data
    let cell_data = data_loader_spatial();
    println!("Number of cells: {}", cell_data.len());
    // // find the similar cells (exact same if any)
    let (similarities, map) = find_similar_close_by_cells(&cell_data);
    let mut count_0_distance = 0;
    for distance in similarities.iter() {
        if distance.2 > (1.0 / 500.0) {
            println!("Cell A id {} loc {:?} total {} Cell B id {} loc {:?} total {} Gene_distance {}", distance.0.cell_id, distance.0.spatial_location, distance.0.total_count, distance.1.cell_id, distance.1.spatial_location, distance.1.total_count, distance.2);
            count_0_distance += 1;
        }
    }
    println!("Distances count: {}", count_0_distance);
    // // check median of map
    let mut lengths: Vec<usize> = map.values().map(|v| v.len()).collect();
    let sum: usize = lengths.iter().sum();
    println!("Average: {}", sum as f64 / lengths.len() as f64);
    lengths.sort_unstable();
    println!("Median: {}", lengths[lengths.len() / 2] as f64);
    let zero_lens = lengths.iter().filter(|&&l| l == 1).count();
    println!("zero neighbour cells {}", zero_lens);

    // // make the initial graph using pet graph
    // make_the_graph(&cell_data, similarities);
    // // go through connected sections and make connection by em

    // // cluster the graph Stoer–Wagner algorithm

}

fn leiden_test() {
    let mut nodes: HashMap<&'static str, usize> = HashMap::new();
    let mut g = Graph::new();
    let edges: &[(&'static str, &'static str, f32)] = &[
        ("Fortran", "C", 0.5),
        ("Fortran", "LISP", 0.3),
        ("Fortran", "MATLAB", 0.6),
        ("C", "C++", 0.9),
        ("C", "Go", 0.6),
        ("LISP", "ML", 0.5),
        ("LISP", "OCaml", 0.2),
        ("LISP", "Haskell", 0.2),
        ("LISP", "Ruby", 0.5),
        ("LISP", "Julia", 0.6),
        ("ML", "OCaml", 0.8),
        ("ML", "Haskell", 0.5),
        ("OCaml", "Haskell", 0.3),
        ("OCaml", "F#", 0.6),
        ("Haskell", "Julia", 0.2),
        ("C++", "Python", 0.32),
        ("C++", "Ruby", 0.2),
        ("C++", "C#", 0.5),
        ("Python", "F#", 0.2),
        ("Python", "Julia", 0.4),
        ("C#", "F#", 0.3),
    ];

    for (from, to, weight) in edges.iter() {
        let from_id = *nodes.entry(from).or_insert_with(|| g.add_node(from));
        let to_id = *nodes.entry(to).or_insert_with(|| g.add_node(to));
        g.add_edge(from_id, to_id, (), *weight);
    }

    let mut optimizer = TrivialModularityOptimizer {
            parallel_scale: 128,
            tol: 1e-11,
        };

        let hierarchy = g.leiden(Some(100), &mut optimizer);
        for (i, node) in hierarchy.node_data_slice().iter().enumerate() {
            println!("community {}:", i);
            node.collect_nodes(&|i| {
                let n = g.node_data_slice()[i];
                println!("     {}", n);
            });
        }
}

fn make_the_graph(cell_data: &Vec<CellData>, similarities: Vec<(&CellData, &CellData, f64)>) {
    // initialize the graph
    // let mut g = UnGraphMap::<(f32, f32), usize>::new();
    // // make the nodes
    // for cell in cell_data {
    //     g.add_node(cell.spatial_location);
    // }
    // // make edges using similarities, distance < 200
    // for distance in similarities.iter() {
    //     if distance.2 < 500.0 {
    //         println!("Cell A id {} loc {:?} total {} Cell B id {} loc {:?} total {} Gene_distance {}", distance.0.cell_id, distance.0.spatial_location, distance.0.total_count, distance.1.cell_id, distance.1.spatial_location, distance.1.total_count, distance.2);
    //         // make edge of weight 100
    //         g.add_edge(distance.0.spatial_location, distance.1.spatial_location, 100);
    //     }
    // }
    // // draw the graph to see
    // //et basic_dot = Dot::new(&g);
    // let dot = Dot::with_attr_getters(
    //     &g,
    //     &[Config::NodeNoLabel], // Optional: hide default node index labels
    //     &|_, _| "".to_string(), // Edge attributes
    //     &|_, node| {
    //         let (x, y) = node.1;
    //         // Graphviz pos uses "x,y!" for fixed positions
    //         format!("pos=\"{},{}!\"", x, y)
    //     },
    // );
    // println!("DOT format:\n{:?}\n", dot);
    // gephi draw
}

fn squared_error(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = *x - *y;
            diff * diff
        })
        .sum()
}


fn find_similar_close_by_cells(cells: &Vec<CellData>) -> (Vec<(&CellData, &CellData, f32)>, HashMap<(isize, isize), Vec<&CellData>>) {
    // change the distance so that no cell is alone
    let radius_to_check = vec![10, 100, 500, 1_000, 2_000, 5_000, 10_000, 15_000, 20_000];
    // squared distance of read count difference in two cells
    let mut gene_distances = Vec::new();
    // hash map for keeping track of close by cells for each coordinate
    let mut close_by_map: HashMap<(isize, isize), Vec<&CellData>> = HashMap::new();
    'radi_loop: for radius in radius_to_check {
        for i in 0..cells.len() {
            let cell_a = &cells[i];
            // multiply by 100 and save it as usize
            let cell_a_x = (cell_a.spatial_location.0 * 100.0) as isize;
            let cell_a_y = (cell_a.spatial_location.1 * 100.0) as isize;
            // add to hashmap cell a
            close_by_map.insert((cell_a_x, cell_a_y), vec![cell_a]);
            for j in (i + 1)..cells.len() {
                let cell_b = &cells[j];
                let cell_b_x = (cell_b.spatial_location.0 * 100.0) as isize;
                let cell_b_y = (cell_b.spatial_location.1 * 100.0) as isize;
                // check if cells are within distance L2
                if ((cell_a_x.abs() - cell_b_x.abs()).pow(2) + (cell_a_y.abs() - cell_b_y.abs()).pow(2)).isqrt()
                > radius {
                    continue;
                }
                // cell b is near spatial coordianates of cell a, add to hashmap
                if let Some(vec_value) = close_by_map.get_mut(&(cell_a_x, cell_a_y)) {
                    vec_value.push(cell_b);
                }
                // check read count vecs font match
                if cell_a.read_counts.len() != cell_b.read_counts.len(){
                    continue;
                }
                // check if within count difference
                if cell_a.total_count.abs_diff(cell_b.total_count) > COUNT_DIFF_FOR_SIMILAR {
                    continue;
                }
                let mse = squared_error(&cell_a.glm_data, &cell_b.glm_data);
                gene_distances.push((cell_a, cell_b, (1.0 / mse)));
            }
        }
        let lengths: Vec<usize> = close_by_map.values().map(|v| v.len()).collect();
        let zero_lens = lengths.iter().filter(|&&l| l == 1).count();
        println!("zero neighbour cells {} for radius {}", zero_lens, radius);
        if zero_lens < (lengths.len() / 15) {
            break 'radi_loop;
        }
        else {
            gene_distances.clear();
            close_by_map.clear();
        }
    }
    
    gene_distances.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    (gene_distances, close_by_map)
}

// load spatial data and populate celldata vec
fn data_loader_spatial() -> Vec<CellData> {
    // for testing load only the cells in a small area 
    let x_limit = 33_000.0; //225 cells when x and y 33_000
    let y_limit = 33_000.0;
    let mut process_cells = vec![];
    println!("Start Coor Data Loading");
    // make a hashmap for look up of spatial locaiton of cell id
    let mut spatial_lookup: HashMap<String, (f32, f32)> = HashMap::new();
    let meta_file = File::open(COOR_FILE).expect("cannot open data file");
    let meta_data_reader = BufReader::new(meta_file);
    let mut x_loc = 0;
    let mut y_loc = 0;
    for (line_index, line) in meta_data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        if line_index == 0 {
            for (value_index, value) in values.iter().enumerate() {
                if *value == "CenterX_global_px" || *value == "xcoord" {
                    x_loc = value_index;
                }
                if *value == "CenterY_global_px" || *value == "ycoord" {
                    y_loc = value_index;
                }
            }
        }
        else {
            let x = values[x_loc].parse().unwrap();
            let y = values[y_loc].parse().unwrap();
            spatial_lookup.insert(values[0].trim().to_string(), (x, y));
        }
    }
    println!("Start GLM PCA Data Loading");
    let mut glm_lookup: HashMap<String, Vec<f32>> = HashMap::new();
    let glm_file = File::open(GLM_FILE).expect("cannot open data file");
    let glm_data_reader = BufReader::new(glm_file);
    for (_line_index, line) in glm_data_reader.lines().enumerate().skip(1) {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        let mut pca_values = vec![];
        let mut cell_id = "".to_string();
        for (value_index, value) in values.iter().enumerate() {
            if value_index == 0 {
                cell_id = value.to_string();
            }
            else {
                pca_values.push(value.to_string().parse::<f32>().unwrap());
            }
        }
        glm_lookup.insert(cell_id, pca_values);
    }
    println!("Start Count Data Loading");
    let data_file = File::open(COUNT_FILE).expect("cannot open data file");
    let data_reader = BufReader::new(data_file);
    let mut all_cell_data = vec![];
    for (line_index, line) in data_reader.lines().enumerate() {
        let line = line.unwrap();
        let values: Vec<&str> = line.split(',').collect();
        // first line has the cell ids
        if line_index == 0 {
            // save the values in a vector, cell id_fov_etc
            for (value_index, value) in values.iter().enumerate() {
                let cell_id = value.trim().to_string();
                if cell_id == "Row" {
                    continue;
                }
                // using the hashmap find the spatial location
                let (spatial_location_x, spatial_location_y) = spatial_lookup.get(&cell_id).cloned().unwrap_or((0.0, 0.0));
                let glm_vec = glm_lookup.get(&cell_id).cloned().unwrap_or(vec![]);
                if (spatial_location_x < x_limit) && (spatial_location_y < y_limit) {
                    process_cells.push(value_index);
                    let temp_cell_data = CellData::new(cell_id, (spatial_location_x, spatial_location_y), glm_vec);
                    all_cell_data.push(temp_cell_data);
                }
            }
            println!("{}", all_cell_data.len());
            continue;
        }
        let mut gene_expressed_by_cells = 0;
        // only use the genes which are expressed by num of cells greater than GENE_CELL_CUTOFF
        for (_cell_index, value) in values.iter().enumerate().skip(1) {
            // convert to u32 and add to cell data
            let read_count = match value.to_string().parse::<u32>() {
                Ok(x) => {x}
                Err(_) => {continue}
            };
            if read_count > 0 {
                gene_expressed_by_cells += 1;
            }
        }
        if gene_expressed_by_cells > GENE_CELL_CUTOFF {
            println!("gene {} passed", line_index);
            let mut real_cell_index = 0;
            for (cell_index, value) in values.iter().enumerate() {
                if !process_cells.contains(&cell_index) {
                    continue;
                }
                // convert to u32 and add to cell data
                let read_count = value.to_string().parse::<u32>().unwrap();
                all_cell_data[real_cell_index].read_counts.push(read_count);
                all_cell_data[real_cell_index].total_count += read_count as usize;
                if read_count > 0 {
                    all_cell_data[real_cell_index].genes_with_count += 1;
                }
                real_cell_index += 1;
            }
        }
    }
    println!("End Data Loading");
    all_cell_data
}

#[derive(Clone)]
struct CellData {
    cell_id: String,
    read_counts: Vec<u32>,
    spatial_location: (f32, f32),
    total_count: usize,
    genes_with_count: usize,
    glm_data: Vec<f32>
}
impl CellData {
    fn new(cell_id: String, spatial_xy: (f32, f32), glm_vec: Vec<f32>) -> CellData {
        CellData{
            cell_id: cell_id,
            read_counts: vec![],
            spatial_location: spatial_xy,
            total_count: 0,
            genes_with_count: 0,
            glm_data: glm_vec
        }
    }
}