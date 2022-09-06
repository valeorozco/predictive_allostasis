def parse_params(path_to_params):
    """
    Input file format:
    csv_filename =  Results/FF/experiment1/testFF
    range_delays = 0.05, 0.5
    learning_rate = 1e-7
    error_ff = itemp_attractor or smooth_error      
    desired_value_av = 0.7
    input_attractor = weighted_output or error
    """
    with open(path_to_params) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    #parse variables 
    csv_filename = lines[0].split("=")[1].strip()
    range_delays = lines[1].split("=")[1].strip().split(",")
    range_delays = [float(i.strip()) for i in range_delays]
    learning_rate = float(lines[2].split("=")[1].strip())
    error_ff =  lines[3].split("=")[1].strip() 
    desired_value_av =  float(lines[4].split("=")[1].strip())
    input_attractor =  lines[5].split("=")[1].strip()

    return {"csv_filename": csv_filename,
            "range_delays" : range_delays,
            "learning_rate": learning_rate,
            "error_ff" : error_ff,      
            "desired_value_av" : desired_value_av,
            "input_attractor": input_attractor
            }

