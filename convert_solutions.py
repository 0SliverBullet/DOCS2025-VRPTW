#!/usr/bin/env python3
"""
Convert solution files to the DOCS2025-VRPTW specified format with route details.
"""

import os
import math
import re
from typing import List, Tuple, Dict

def read_problem_data(data_file_path: str) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int]]:
    """
    Read problem data file and extract coordinates, demands, service times, and time windows.
    Apply DIMACS scaling (multiply coordinates and times by 10) to match the solver's format.
    Returns: (coordinates, demands, service_times, ready_times, due_dates)
    """
    coordinates = {}  # customer_id -> (x, y)
    demands = {}      # customer_id -> demand
    service_times = {} # customer_id -> service_time
    ready_times = {}  # customer_id -> ready_time (time window start)
    due_dates = {}    # customer_id -> due_date (time window end)
    
    with open(data_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find the start of customer data
    customer_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("CUST NO."):
            customer_start = i + 2  # Skip header line and empty line
            break
    
    if customer_start == -1:
        raise ValueError("Could not find customer data in file")
    
    # Read customer data
    # Format: CUST NO.  XCOORD.  YCOORD.  DEMAND  READY TIME  DUE DATE  SERVICE TIME
    for i in range(customer_start, len(lines)):
        line = lines[i].strip()
        if not line:
            break
            
        parts = line.split()
        if len(parts) >= 7:
            cust_no = int(parts[0])
            x_coord = float(parts[1]) * 10  # Apply DIMACS scaling
            y_coord = float(parts[2]) * 10  # Apply DIMACS scaling
            demand = int(parts[3])
            ready_time = int(parts[4]) * 10  # Apply DIMACS scaling for time window start
            due_date = int(parts[5]) * 10    # Apply DIMACS scaling for time window end
            service_time = int(parts[6]) * 10  # Apply DIMACS scaling for service time
            
            coordinates[cust_no] = (x_coord, y_coord)
            demands[cust_no] = demand
            service_times[cust_no] = service_time
            ready_times[cust_no] = ready_time
            due_dates[cust_no] = due_date
    
    return coordinates, demands, service_times, ready_times, due_dates

def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> int:
    """Calculate Euclidean distance between two coordinates, return as int (truncated)."""
    return int(math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2))

def parse_solution_file(solution_file_path: str) -> Tuple[List[List[int]], float]:
    """
    Parse solution file and extract routes and total distance.
    Returns: (routes, total_distance)
    """
    routes = []
    total_distance = 0.0
    
    with open(solution_file_path, 'r') as file:
        content = file.read()
    
    # Extract total distance
    distance_match = re.search(r'distance:\s*(\d+(?:\.\d+)?)', content)
    if distance_match:
        total_distance = float(distance_match.group(1))
    
    # Extract routes
    routes_section = False
    for line in content.split('\n'):
        line = line.strip()
        if line == "Routes" or line == "------":
            routes_section = True
            continue
        
        if routes_section and line.startswith("Route #"):
            # Extract customer sequence from route line
            route_match = re.search(r'Route #\d+:\s*(.+)', line)
            if route_match:
                customer_sequence = route_match.group(1).strip()
                if customer_sequence:
                    # Parse customer numbers
                    customers = [int(x) for x in customer_sequence.split()]
                    routes.append(customers)
    
    return routes, total_distance

def calculate_route_metrics(route: List[int], coordinates: Dict[int, Tuple[float, float]], 
                          demands: Dict[int, int], service_times: Dict[int, int],
                          ready_times: Dict[int, int], due_dates: Dict[int, int]) -> Tuple[float, int, float]:
    """
    Calculate distance, total demand, and total time for a single route.
    Returns: (distance, total_demand, total_time)
    """
    if not route:
        return 0.0, 0, 0.0
    
    # Add depot (0) at start and end
    full_route = [0] + route + [0]
    
    # Calculate total distance
    total_distance = 0.0
    for i in range(len(full_route) - 1):
        from_customer = full_route[i]
        to_customer = full_route[i + 1]
        coord_from = coordinates[from_customer]
        coord_to = coordinates[to_customer]
        total_distance += calculate_distance(coord_from, coord_to)
    
    # Calculate total demand (only for customers, not depot)
    total_demand = sum(demands[customer] for customer in route)
    
    # Calculate total time (travel time + service time + waiting for time window start)
    total_time = 0.0
    current_time = 0.0
    full_route = [0] + route + [0]
    for i in range(len(full_route) - 1):
        from_customer = full_route[i]
        to_customer = full_route[i + 1]
        coord_from = coordinates[from_customer]
        coord_to = coordinates[to_customer]
        travel_time = calculate_distance(coord_from, coord_to)
        current_time += travel_time

        # For all customers except depot, add waiting time if arrival is before ready time
        if i + 1 < len(full_route) - 1:  # skip depot at end
            ready_time = ready_times.get(to_customer, 0)  # Get actual ready time from data
            if current_time < ready_time:
                current_time = ready_time  # Wait until time window opens

            current_time += service_times[to_customer]
        else:
            # At depot, no service time
            pass

    total_time = current_time
    
    return total_distance, total_demand, total_time

def convert_solution_file(solution_file_path: str, data_file_path: str, output_file_path: str):
    """Convert a single solution file to the specified format."""
    
    # Read problem data
    coordinates, demands, service_times, ready_times, due_dates = read_problem_data(data_file_path)
    
    # Parse solution file
    routes, total_distance = parse_solution_file(solution_file_path)
    
    # Generate output
    output_lines = []
    calculated_total_distance = 0.0
    
    for i, route in enumerate(routes, 1):
        if not route:
            continue
            
        # Calculate route metrics
        route_distance, route_demand, route_time = calculate_route_metrics(
            route, coordinates, demands, service_times, ready_times, due_dates
        )
        calculated_total_distance += route_distance
        
        # Format route output
        customers_str = " -> ".join(["0"] + [str(c) for c in route] + ["0"])
        
        # Convert back from DIMACS format for display (divide by 10)
        display_distance = route_distance / 10
        display_time = route_time / 10
        
        output_lines.append(f"Route for Vehicle {i}:")
        output_lines.append(f"  Customers: {customers_str}")
        output_lines.append(f"  Distance: {display_distance:.1f}")
        output_lines.append(f"  Total Demand: {route_demand}")
        output_lines.append(f"  Total Time: {display_time:.1f}")
        output_lines.append("")  # Empty line between routes
    
    # Add total distance
    # Use the calculated total distance (more accurate than parsing from file)
    # Convert back from DIMACS format for display (divide by 10)
    display_total_distance = calculated_total_distance / 10
    output_lines.append(f"Total Distance for All Vehicles: {display_total_distance:.1f}")

    # Also compute and output total time for all vehicles
    calculated_total_time = 0.0
    for i, route in enumerate(routes, 1):
        if not route:
            continue
        _, _, route_time = calculate_route_metrics(
            route, coordinates, demands, service_times, ready_times, due_dates
        )
        calculated_total_time += route_time
    display_total_time = calculated_total_time / 10
    output_lines.append(f"Total Time for All Vehicles: {display_total_time:.1f}")
    
    # Write output file
    with open(output_file_path, 'w') as file:
        file.write('\n'.join(output_lines))
    
    print(f"Converted {solution_file_path} -> {output_file_path}")

def main():
    """Main function to convert all solution files."""
    
    # Define paths
    solutions_dir = "solutions/unformatted_solutions"
    output_dir = "solutions/formatted_solutions"
    data_200_dir = "data/homberger_200_customer_instances"
    data_800_dir = "data/homberger_800_customer_instances"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all solution files
    solution_files = [f for f in os.listdir(solutions_dir) if f.endswith('.txt')]
    
    for solution_file in solution_files:
        solution_path = os.path.join(solutions_dir, solution_file)
        
        # Determine corresponding data file
        base_name = solution_file.replace('.txt', '')
        data_file = f"{base_name}.TXT"
        
        # Check in 200 customer instances first, then 800
        data_path = os.path.join(data_200_dir, data_file)
        if not os.path.exists(data_path):
            data_path = os.path.join(data_800_dir, data_file)
        
        if not os.path.exists(data_path):
            print(f"Warning: Could not find data file for {solution_file}")
            continue
        
        # Convert solution
        output_path = os.path.join(output_dir, f"formatted_{solution_file}")
        try:
            convert_solution_file(solution_path, data_path, output_path)
        except Exception as e:
            print(f"Error converting {solution_file}: {e}")

if __name__ == "__main__":
    main() 