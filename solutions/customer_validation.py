import re

file_path = 'results0806/RC1_4_1.txt'
customer_count = 400

def parse_and_check_customers(file_content):
    """
    Parses customer numbers from vehicle routes in the file content,
    and checks if all customers from 1 to customer_count are present exactly once.
    """
    # Clean up the file content to handle multiline customer lists
    # This regex handles cases where a line ends with '->' and the next line continues the list.
    cleaned_content = re.sub(r'->\s*\n\s*', '-> ', file_content)
    cleaned_content = cleaned_content.replace('Route \nfor Vehicle', 'Route for Vehicle')
    cleaned_content = cleaned_content.replace('\n', ' ')
    
    # Extract all customer numbers
    all_customers = []
    # Find all occurrences of "Customers: 0 -> ... -> 0"
    customer_lists = re.findall(r'Customers: (.*?)(?=Distance:)', cleaned_content)

    for cust_list_str in customer_lists:
        customers = [int(c.strip()) for c in cust_list_str.split('->') if c.strip() and c.strip() != '0']
        all_customers.extend(customers)

    # Check for coverage, duplicates, and missing values
    if not all_customers:
        return "没有找到任何客户点。"

    # Check for duplicates by comparing list length to set length
    if len(all_customers) != len(set(all_customers)):
        # Find and report the duplicates
        seen = set()
        duplicates = sorted([c for c in all_customers if c in seen or seen.add(c)])
        return f"存在重复的点: {duplicates}"

    # Check for missing customers
    customer_set = set(all_customers)
    expected_customers = set(range(1, customer_count + 1))
    missing_customers = sorted(list(expected_customers - customer_set))

    if missing_customers:
        return f"缺少点: {missing_customers}"
        
    # Check for any customers outside the expected range
    unexpected_customers = sorted(list(customer_set - expected_customers))
    if unexpected_customers:
        return f"存在范围外的点: {unexpected_customers}"

    return "所有点都包含在内，没有重复，没有遗漏，每个点都出现且仅出现一次。"

# Read the content of the uploaded file
with open(file_path, 'r') as f:
    file_content = f.read()

# Perform the check and print the result
result = parse_and_check_customers(file_content)
print(result)