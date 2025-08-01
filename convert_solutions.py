#!/usr/bin/env python3
"""
Convert solution files to the DOCS2025-VRPTW specified format with route details.
"""

import os
import math
import re
from typing import List, Tuple, Dict

def read_problem_data(data_file_path: str) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, int], Dict[int, int], Dict[int, int], Dict[int, int], int, int, int]:
    """
    Read problem data file and extract coordinates, demands, service times, time windows, and vehicle info.
    Apply DOCS scaling (multiply coordinates and times by 100) to match the solver's format.
    Returns: (coordinates, demands, service_times, ready_times, due_dates, vehicle_capacity, depot_ready_time, depot_due_time)
    """
    coordinates = {}  # customer_id -> (x, y)
    demands = {}      # customer_id -> demand
    service_times = {} # customer_id -> service_time
    ready_times = {}  # customer_id -> ready_time (time window start)
    due_dates = {}    # customer_id -> due_date (time window end)
    vehicle_capacity = 0
    depot_ready_time = 0
    depot_due_time = 0
    
    with open(data_file_path, 'r') as file:
        lines = file.readlines()
    
    # Find and read vehicle information
    vehicle_section = False
    for i, line in enumerate(lines):
        if line.strip().startswith("VEHICLE"):
            vehicle_section = True
            continue
        if vehicle_section and line.strip().startswith("NUMBER"):
            # Next line should contain vehicle count and capacity
            if i + 1 < len(lines):
                vehicle_data = lines[i + 1].strip().split()
                if len(vehicle_data) >= 2:
                    vehicle_capacity = int(vehicle_data[1])
            break
    
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
            x_coord = float(parts[1]) * 100  # Apply DOCS scaling
            y_coord = float(parts[2]) * 100  # Apply DOCS scaling
            demand = int(parts[3])
            ready_time = int(parts[4]) * 100  # Apply DOCS scaling for time window start
            due_date = int(parts[5]) * 100    # Apply DOCS scaling for time window end
            service_time = int(parts[6]) * 100  # Apply DOCS scaling for service time
            
            coordinates[cust_no] = (x_coord, y_coord)
            demands[cust_no] = demand
            service_times[cust_no] = service_time
            ready_times[cust_no] = ready_time
            due_dates[cust_no] = due_date
            
            # Store depot information (customer 0)
            if cust_no == 0:
                depot_ready_time = ready_time
                depot_due_time = due_date
    
    return coordinates, demands, service_times, ready_times, due_dates, vehicle_capacity, depot_ready_time, depot_due_time

# def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> int:
#     """Calculate Euclidean distance between two coordinates, return as int (truncated)."""
#     return int(math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2))

def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> int:
    """计算两个坐标点之间的欧氏距离，结果四舍五入后返回整数。"""
    # 首先，计算出带有小数的精确距离
    distance = math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
    
    # 然后，使用 round() 函数进行四舍五入，并返回结果
    return round(distance)


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
                          ready_times: Dict[int, int], due_dates: Dict[int, int],
                          vehicle_capacity: int, depot_due_time: int) -> Tuple[float, int, float, List[str]]:
    """
    Calculate distance, total demand, total time for a single route and check constraints.
    Returns: (distance, total_demand, total_time, constraint_violations)
    """
    if not route:
        return 0.0, 0, 0.0, []
    
    # Add depot (0) at start and end
    full_route = [0] + route + [0]
    constraint_violations = []
    
    # Calculate total distance
    total_distance = 0.0
    for i in range(len(full_route) - 1):
        from_customer = full_route[i]
        to_customer = full_route[i + 1]
        coord_from = coordinates[from_customer]
        coord_to = coordinates[to_customer]
        total_distance += calculate_distance(coord_from, coord_to)
    
    # Calculate total demand and check capacity constraint
    total_demand = sum(demands[customer] for customer in route)
    if total_demand > vehicle_capacity:
        constraint_violations.append(f"Capacity violation: {total_demand} > {vehicle_capacity}")
    
    # Calculate total time and check time window constraints
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

        # For all customers except depot, check time window constraints
        if i + 1 < len(full_route) - 1:  # skip depot at end
            ready_time = ready_times.get(to_customer, 0)
            due_time = due_dates.get(to_customer, float('inf'))
            
            # Check if arrival is too late (after due time)
            if current_time > due_time:
                constraint_violations.append(f"Time window violation at customer {to_customer}: arrival {current_time/100:.2f} > due time {due_time/100:.2f}")

            # Wait if arrival is before ready time
            if current_time < ready_time:
                current_time = ready_time  # Wait until time window opens

            current_time += service_times[to_customer]
        else:
            # At depot, check if return time exceeds depot's latest time
            if current_time > depot_due_time:
                constraint_violations.append(f"Depot time violation: return time {current_time/100:.2f} > depot due time {depot_due_time/100:.2f}")

    total_time = current_time
    
    return total_distance, total_demand, total_time, constraint_violations

def convert_solution_file(solution_file_path: str, data_file_path: str, output_file_path: str):
    """Convert a single solution file to the specified format."""
    
    # Read problem data
    coordinates, demands, service_times, ready_times, due_dates, vehicle_capacity, depot_ready_time, depot_due_time = read_problem_data(data_file_path)
    
    # Parse solution file
    routes, total_distance = parse_solution_file(solution_file_path)
    
    # Generate output
    output_lines = []
    calculated_total_distance = 0.0
    total_violations = 0
    
    for i, route in enumerate(routes, 1):
        if not route:
            continue
            
        # Calculate route metrics and check constraints
        route_distance, route_demand, route_time, violations = calculate_route_metrics(
            route, coordinates, demands, service_times, ready_times, due_dates, vehicle_capacity, depot_due_time
        )
        calculated_total_distance += route_distance
        
        # Format route output
        customers_str = " -> ".join(["0"] + [str(c) for c in route] + ["0"])

        # Convert back from DOCS format for display (divide by 100)
        display_distance = route_distance / 100
        display_time = route_time / 100
        
        output_lines.append(f"Route for Vehicle {i}:")
        output_lines.append(f"  Customers: {customers_str}")
        output_lines.append(f"  Distance: {display_distance:.2f}")
        output_lines.append(f"  Total Demand: {route_demand}")
        output_lines.append(f"  Total Time: {display_time:.2f}")
        
        # Add constraint violation information
        if violations:
            total_violations += len(violations)
            output_lines.append(f"  *** CONSTRAINT VIOLATIONS ***")
            for violation in violations:
                output_lines.append(f"    - {violation}")
        else:
            # output_lines.append(f"  Status: FEASIBLE")
            pass
        
        output_lines.append("")  # Empty line between routes
    
    # Add total distance
    # Use the calculated total distance (more accurate than parsing from file)
    # Convert back from DOCS format for display (divide by 100)
    display_total_distance = calculated_total_distance / 100
    output_lines.append(f"Total Distance for All Vehicles: {display_total_distance:.2f}")

    # Also compute and output total time for all vehicles
    calculated_total_time = 0.0
    for i, route in enumerate(routes, 1):
        if not route:
            continue
        _, _, route_time, _ = calculate_route_metrics(
            route, coordinates, demands, service_times, ready_times, due_dates, vehicle_capacity, depot_due_time
        )
        calculated_total_time += route_time
    display_total_time = calculated_total_time / 100
    output_lines.append(f"Total Time for All Vehicles: {display_total_time:.2f}")
    
    # Add overall feasibility summary
    output_lines.append("")
    if total_violations == 0:
        # output_lines.append("SOLUTION STATUS: FEASIBLE")
        pass
    else:
        output_lines.append(f"SOLUTION STATUS: INFEASIBLE ({total_violations} constraint violations)")
    
    # Write output file
    with open(output_file_path, 'w') as file:
        file.write('\n'.join(output_lines))
    
    print(f"Converted {solution_file_path} -> {output_file_path} ({total_violations} violations)")

def main():
    """Main function to convert all solution files."""
    
    # Define paths
    solutions_dir = "solutions/unformatted_solutions/results0730"
    output_dir = "solutions/results0730"
    data_200_dir = "data/homberger_200_customer_instances"
    data_400_dir = "data/homberger_400_customer_instances"
    
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
            data_path = os.path.join(data_400_dir, data_file)
        
        if not os.path.exists(data_path):
            print(f"Warning: Could not find data file for {solution_file}")
            continue
        
        # Convert solution
        output_path = os.path.join(output_dir, f"{solution_file}")
        try:
            convert_solution_file(solution_path, data_path, output_path)
        except Exception as e:
            print(f"Error converting {solution_file}: {e}")

if __name__ == "__main__":
    main() 