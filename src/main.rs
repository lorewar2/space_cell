mod leiden_alg;

use std::cell;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::io::Write;
use std::collections::HashMap;
use std::cmp::Reverse;
use petgraph::graphmap::UnGraphMap;
use leiden_alg::{Graph, TrivialModularityOptimizer};

const COUNT_FILE: &'static str = "./data/visium_count.csv";
const COOR_FILE: &'static str = "./data/visium_coor.csv";
const GLM_FILE: &'static str = "./data/visium_glm.csv";
const GENE_CELL_CUTOFF: usize = 2000;
const COUNT_DIFF_FOR_SIMILAR: usize = 500;

fn main() {
    //_leiden_test();
    make_connections_and_run_clustering();
}

fn make_connections_and_run_clustering() {
    // load the data
    let cell_data = data_loader_spatial();
    println!("Number of cells: {}", cell_data.len());
    let cell_close_by = find_close_cells_l1(&cell_data);
    // // find the similar cells (exact same if any)
    let similarities = loss_distance_similarity_calculator(&cell_data, &cell_close_by);

    // better graph
    //let new_cells = merge_similar_cells_connect_merged_with_loss(similarities, &cell_data);

    // connect merged
   // let similarities = merged_connection(&new_cells);
    //let (similarities, _map) = find_similar_close_by_cells(&cell_data);
    // // leiden run
    save_graph_for_leiden(similarities);
}

fn merged_connection<'a>(cells: &'a Vec<CellData>) -> Vec<(&'a CellData, &'a CellData, f32)> {
    let mut gene_distances = vec![];
    // first try with glm pca loss
    for i in 0..cells.len() {
        let cell_a = &cells[i];
        for j in (i + 1)..cells.len() {
            let cell_b = &cells[j];
            let mse = squared_error(&cell_a.glm_data, &cell_b.glm_data);
            gene_distances.push((cell_a, cell_b, (1.0 / mse)));
        }
    }
    println!("new connections {}", gene_distances.len());
    gene_distances
}

fn merge_similar_cells_connect_merged_with_loss(similarities: Vec<(&CellData, &CellData, f32)>, cells: &Vec<CellData>) -> Vec<CellData> {
    // union join on the previous connection to get the similar blocks
    println!("Connections {}", similarities.len());
    let mut cell_to_cluster: HashMap<usize, usize> = HashMap::new();
    let mut cluster_num = 0;
    for arr in &similarities {
        let a = arr.0.cell_index;
        let b = arr.1.cell_index;
        let a_cluster = cell_to_cluster.get(&a).copied();
        let b_cluster = cell_to_cluster.get(&b).copied();
        match (a_cluster, b_cluster) {
            // both in clusters
            (Some(c1), Some(c2)) => {
                if c1 != c2 {
                    // merge c2 into c1
                    for val in cell_to_cluster.values_mut() {
                        if *val == c2 {
                            *val = c1;
                        }
                    }
                }
            }
            // none in clusters
            (None, None) => {
                cell_to_cluster.insert(a, cluster_num);
                cell_to_cluster.insert(b, cluster_num);
                cluster_num += 1;
            }
            // a in cluster, b not
            (Some(c), None) => {
                cell_to_cluster.insert(b, c);
            }
            // b in cluster, a not
            (None, Some(c)) => {
                cell_to_cluster.insert(a, c);
            }
        }
    }
    
    // cluster_id -> count
    // cell_id to hashmap
    let mut cluster_cells = vec![vec![]; similarities.len()];
    let mut cell_index_to_cell: HashMap<usize, &CellData> = HashMap::new();
    for cell in cells {
        cell_index_to_cell.insert(cell.cell_index, &cell);
    }
    for (cell_index, cluster) in cell_to_cluster.iter() {
        cluster_cells[*cluster].push(cell_index_to_cell[cell_index]);
    }
    // sort the cluster cells by size of each cluster
    cluster_cells.sort_by_key(|inner_vec| Reverse(inner_vec.len()));
    // only use the best 50 clusters
    cluster_cells.truncate(50);
    // print results
    let mut ignored_cells = 0;
    let mut assigned_cells = 0;
    for (index, cluster) in cluster_cells.iter().enumerate() {
        if cluster.len() > 10 {
            println!("Cluster {} has {} cells", index, cluster.len());
            assigned_cells += cluster.len();
        }
        else {
            ignored_cells += cluster.len();
        }
    }
    println!("unassigned {} assigned {} ignored {} total {}", cells.len() - assigned_cells, assigned_cells, ignored_cells, cells.len());
    // calculate the average
    let mut new_cell_vec = vec![];
    // make a new cell for the whole cluster
    for (clus_index, cluster) in cluster_cells.iter().enumerate() {
        let number_of_cells_in_cluster = cluster.len();
        let mut x_location_total = 0.0;
        let mut y_location_total = 0.0;
        let mut count_total_vec = vec![];
        let mut glm_pca_total_vec = vec![];
        
        for cell in cluster {
            // initialize the vectors
            if count_total_vec.len() == 0 {
                count_total_vec = vec![0; cell.read_counts.len()];
            }
            if glm_pca_total_vec.len() == 0 {
                glm_pca_total_vec = vec![0.0; cell.glm_data.len()];
            }
            // get count 
            for (index, read) in cell.read_counts.iter().enumerate() {
                count_total_vec[index] += read;
            }
            // get glm pca 
            for (index, glm) in cell.glm_data.iter().enumerate() {
                glm_pca_total_vec[index] += glm;
            }
            // get location 
            x_location_total += cell.spatial_location.0;
            y_location_total += cell.spatial_location.1;
        }
        // get average and initialize a cell
        for value in &mut count_total_vec {
            *value = *value / (number_of_cells_in_cluster as u32);
        }
        for value in &mut glm_pca_total_vec {
            *value = *value / (number_of_cells_in_cluster as f32);
        }
        x_location_total = x_location_total / number_of_cells_in_cluster as f32;
        y_location_total = y_location_total / number_of_cells_in_cluster as f32;

        let new_cell = CellData::new(clus_index, clus_index.to_string(), (x_location_total, y_location_total), glm_pca_total_vec);
        //println!("{},{}", x_location_total, y_location_total);
        new_cell_vec.push(new_cell);
    }
    println!("len of new cell vec {}", new_cell_vec.len());
    return new_cell_vec;
}

fn loss_distance_similarity_calculator<'a>(cells: &'a Vec<CellData>, cell_close_by: &'a Vec<Vec<&'a CellData>>) -> Vec<(&'a CellData, &'a CellData, f32)> {
    let distance_weight = 0.0;
    let glm_simi_weight = 1.0;
    let raw_count_weight = 0.0;
    let mut loss_between_cells: Vec<(&CellData, &CellData, f32)> = Vec::new();
    for (origin_cell, close_cell_vec) in cells.iter().zip(cell_close_by) {
        for close_cell in close_cell_vec {
            if close_cell.cell_id == origin_cell.cell_id {
                continue;
            }
            // calculate the distance between target and close cell
            let (x_origin, y_origin) = origin_cell.spatial_location;
            let (x_close, y_close) = close_cell.spatial_location;
            let l2_distance = euclidean_distance(x_origin * 100.0, y_origin * 100.0, x_close * 100.0, y_close * 100.0);
            // calculate the raw count similarity
            let count_difference =  origin_cell.total_count.abs_diff(close_cell.total_count) as f32;
            // calculate the similarity between target and close cell
            let glm_similarity = squared_error(&origin_cell.glm_data, &close_cell.glm_data);
            // calculate the loss using distance and similarity
            let loss = (distance_weight * l2_distance) + (glm_simi_weight * glm_similarity) + (count_difference * raw_count_weight);
            // inverse loss because leiden weights are high = well connected
            let inverse_loss = 1.0 / loss;
            println!("l2 distance {} glm dist {} raw_count {} loss {} inverse_loss {}", l2_distance, glm_similarity, count_difference, loss, inverse_loss);
            // dont use if not within threshold
            if count_difference < 300.0 && glm_similarity < 2_100.0 && l2_distance < 30_000.0 { // 20 1500 5000
                // append to loss between cells
                loss_between_cells.push((origin_cell, close_cell, inverse_loss));
            }
        }
    }
    loss_between_cells
}

fn euclidean_distance(x1: f32, y1: f32, x2: f32, y2: f32) -> f32 {
    ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
}

fn find_close_cells_l1(cells: &Vec<CellData>) -> Vec<Vec<&CellData>> {
    // index hash map, match index to cell, and generate spatial map
    let mut index_cell_map: HashMap<usize, &CellData> = HashMap::new();
    let mut spatial_map: HashMap<(isize, isize), &CellData> = HashMap::new();
    for cell in &*cells {
        index_cell_map.insert(cell.cell_index, cell);
        let x = (cell.spatial_location.0 * 100.0) as isize;
        let y = (cell.spatial_location.1 * 100.0) as isize;
        spatial_map.insert((x, y), cell);
    }
    let mut all_values = vec![];
    let mut cell_closeby_cells = vec![];
    // go through each cell and generate the close 100 cell list with distances
    for cell in &*cells {
        let x_main = (cell.spatial_location.0 * 100.0) as isize;
        let y_main = (cell.spatial_location.1 * 100.0) as isize;
        let mut radius = 10_000; // check l1 distance for efficiency
        let mut temp_list: Vec<&CellData>; // id distance
        'radi_loop: loop {
            temp_list = vec![];
            let x_main_radius = (x_main - radius, x_main + radius);
            let y_main_radius = (y_main - radius, y_main + radius);
            // go through all the locations
            for (x_key, y_key) in spatial_map.keys() {
                if (*x_key  < x_main_radius.0) || (*x_key > x_main_radius.1) {
                    // not within x range
                    continue;
                }
                if (*y_key < y_main_radius.0)  || (*y_key > y_main_radius.1) {
                    // not within y range
                    continue;
                }
                let key_cell_id = spatial_map.get(&(*x_key, *y_key)).unwrap();
                // if reached here, should be within range in l1
                //println!("cell index {},  ({}, {}), close by withing l1 {} to cell index {} ({}, {})", cell.cell_index, x_main, y_main, radius, key_cell_id, x_key, y_key);
                temp_list.push(*key_cell_id);
            }
            // loop until we find atleast 100 entries
            println!("cell index {},  ({}, {}), radius {} number of cells {}", cell.cell_index, x_main, y_main, radius, temp_list.len());
            // make sure doesnt go over either
            // under
            if temp_list.len() < 100 {
                radius = radius * 5 / 4;
            }
            else {
                all_values.push(temp_list.len());
                // found the required number of cells, save the cell keys
                break 'radi_loop;
            }
        }
        cell_closeby_cells.push(temp_list);
        // check counts
        all_values.sort_unstable();
        println!("Max count {} Median {} Min {}", all_values[all_values.len() - 1], all_values[all_values.len() / 2], all_values[0]);
    }
    cell_closeby_cells
}

fn save_graph_for_leiden(similarities: Vec<(&CellData, &CellData, f32)>) {
    let mut g = UnGraphMap::new();
    let mut node_id_map: HashMap<String, usize> = HashMap::new();  // string → integer
    let mut id_to_label: HashMap<usize, String> = HashMap::new();  // integer → string
    let mut next_id = 0;
    for distance in similarities.iter() {
        let from_name = distance.0.cell_id.clone();
        let to_name   = distance.1.cell_id.clone();
        // assign integer IDs
        let from_id = *node_id_map.entry(from_name.clone())
            .or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id_to_label.insert(id, from_name.clone());  // map ID → label
                id
            });
        let to_id = *node_id_map.entry(to_name.clone())
            .or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id_to_label.insert(id, to_name.clone());    // map ID → label
                id
            });
        // UnGraphMap automatically inserts nodes if missing
        g.add_edge(from_id, to_id, distance.2);
    }
    //save graph and run in python for now
    save_graphml(&g, &id_to_label, "cere.graphml");
}

fn save_graphml(g: &UnGraphMap<usize, f32>, id_to_label: &HashMap<usize, String>, path: &str) {
    let mut file = File::create(path).expect("Failed to create GraphML file");
    writeln!(file, r#"<?xml version="1.0" encoding="UTF-8"?>"#).unwrap();
    writeln!(file, r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns">"#).unwrap();
    // Node label key
    writeln!(
        file,
        r#"<key id="label" for="node" attr.name="label" attr.type="string"/>"#
    ).unwrap();
    // Edge weight key
    writeln!(
        file,
        r#"<key id="weight" for="edge" attr.name="weight" attr.type="double"/>"#
    ).unwrap();
    writeln!(file, r#"<graph edgedefault="undirected">"#).unwrap();
    for node in g.nodes() {
        let label = id_to_label.get(&node).unwrap();
        writeln!(
            file,
            r#"<node id="n{}"><data key="label">{}</data></node>"#,
            node, label
        ).unwrap();
    }
    // Track written edges to avoid duplicates in undirected graph
    let mut seen_edges = std::collections::HashSet::new();
    for edge in g.all_edges() {
        let (a, b, weight) = edge;
        let key = if a < b { (a, b) } else { (b, a) };
        if seen_edges.insert(key) {
            writeln!(
                file,
                r#"<edge source="n{}" target="n{}"><data key="weight">{}</data></edge>"#,
                a, b, weight
            ).unwrap();
        }
    }
    writeln!(file, "</graph>").unwrap();
    writeln!(file, "</graphml>").unwrap();
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
    let radius_to_check = vec![500, 1_000, 2_000, 5_000, 10_000, 15_000, 20_000];
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
                if cell_a.read_counts.len() != cell_b.read_counts.len() {
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
        if zero_lens < (lengths.len() / 20) {
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
    let mut x_loc = 4;
    let mut y_loc = 5;
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
            println!("{} {}", value_index, value);
            if value_index == 0 || value_index == 1 { // change this to match the dataset
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
    let mut cell_index = 0;
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
                    let temp_cell_data = CellData::new(cell_index, cell_id, (spatial_location_x, spatial_location_y), glm_vec);
                    all_cell_data.push(temp_cell_data);
                    cell_index += 1;
                }
            }
            println!("{}", all_cell_data.len());
            continue;
        }
        let mut gene_expressed_by_cells = 0;
        // only use the genes which are expressed by num of cells greater than GENE_CELL_CUTOFF
        for (_cell_index, value) in values.iter().enumerate().skip(1) {
            // convert to u32 and add to cell data
            let read_count = match value.to_string().parse::<f32>() {
                Ok(x) => {x as u32}
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
                let read_count = value.to_string().parse::<f32>().unwrap() as u32;
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

fn _leiden_test() {
    // for debuggin leiden, ai generated crap doesnt work
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
    println!("number of nodes in the initial graph: {}", g._nodes.len());
    let mut optimizer = TrivialModularityOptimizer {tol: 1e-11};
    let hierarchy = g.leiden(Some(10), &mut optimizer);

    for (i, node) in hierarchy.node_data_slice().iter().enumerate() {
        println!("community {}:", i);
        node.collect_nodes(&|i| {
            let n = g.node_data_slice()[i];
            println!("     {}", n);
        });
    }
}

#[derive(Clone)]
struct CellData {
    cell_index: usize,
    cell_id: String,
    read_counts: Vec<u32>,
    spatial_location: (f32, f32),
    total_count: usize,
    genes_with_count: usize,
    glm_data: Vec<f32>,
    close_cells: Vec<(usize, isize)>, // (cell index, distance)
}
impl CellData {
    fn new(cell_index: usize , cell_id: String, spatial_xy: (f32, f32), glm_vec: Vec<f32>) -> CellData {
        CellData{
            cell_index: cell_index,
            cell_id: cell_id,
            read_counts: vec![],
            spatial_location: spatial_xy,
            total_count: 0,
            genes_with_count: 0,
            glm_data: glm_vec,
            close_cells: vec![]
        }
    }
}