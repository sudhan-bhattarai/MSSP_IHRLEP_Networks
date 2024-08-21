import math
import numpy as np
import copy


def json_export_conversion(file, name=None):
    def convert(val):
        if type(val) is str:
            return val
        else:
            return str(val)
    if name == "pi_mssp":
        converted = {
            convert(t): {
                convert(s): {
                    convert(s_): q for s_, q in d.items()
                    } for s, d in d_dict.items()
                } for t, d_dict in file.items()
            }
    elif name in ["test_samples_from_tree", "in_sample_from_tree_2ssp"]:
        converted = {k: [list(t) for t in v] for k, v in file.items()}
        # converted = {
        #     convert(s): convert(val)
        #     for s, val in file.items()
        #     }
    else:
        converted = {
            convert(t): {convert(s): d for s, d in d_dict.items()}
            for t, d_dict in file.items()
            }
    return converted


def json_import_conversion(obj):
    """
    Convert the 'numeric' keys of json stored as 'string' to 'int' or 'float'
    obj = dict
    """
    def try_convert_to_number(value):
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                try:
                    return eval(value)
                except NameError:
                    return value

    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            converted_key = try_convert_to_number(key)
            if converted_key is not None:
                result[converted_key] = json_import_conversion(value)
            else:
                result[key] = json_import_conversion(value)
        return result
    elif isinstance(obj, list):
        return [json_import_conversion(item) for item in obj]
    else:
        return obj


def distMiles(p1, p2):
    """ 
    Given two GIS points, compute distance in miles 
    """
    dx = (p1[0] - p2[0]) * 54.6  # long degree to miles
    dy = (p1[1] - p2[1]) * 69.0  # lat degree to miles
    return math.sqrt(dx**2 + dy**2)


def dist_matrix(pos_1, pos_2):
    """
    Euclidean distance matrix between facilities 
    pos_1 := GIS of facility 1 = numpy array of shape m * 2
    pos_2 := GIS of facility 2 = numpy array of shape n * 2
    """
    m = pos_1.shape[0]
    n = pos_2.shape[0]
    matrix = np.zeros([m, n])
    for i in range(m):
        for j in range(n):
            dx = (pos_1[i, 0] - pos_2[j, 0]) * 54.6  # long degree to miles
            dy = (pos_1[i, 1] - pos_2[j, 1]) * 69.0  # lat degree to miles
            matrix[i, j] = math.sqrt(dx**2 + dy**2)
    return matrix


def get_hurr_cat(wind_speed):
    """
    For a given wind speed,
    return the respective hurricane category (SS scale)
    """
    cat_scale = {0:73, 1:95, 2:110, 3:129, 4:156, 5:1000}
    for cat, speed in cat_scale.items():
        if wind_speed <= speed:
            return int(cat)
        

def transform_gis_random_landfall(data, t, xi):
    """ Transform along and cross errors to long, lat.

    data : class
        Initial data class.
    t : int
        Period of the Hurricane position.
    xi : tuple
        Along and cross forecast errors at 't'.
    """
    point1 = np.array([data.FORE.X[t], data.FORE.Y[t]])
    if t == 0:
        return point1.tolist()
    point2 = np.array([data.FORE.X[t-1], data.FORE.Y[t-1]])
    miles_convert = np.array([data.Xmiles, data.Ymiles])
    vector = (point1 - point2)
    vector1 = vector * miles_convert
    # Shift point according to "along" error
    vector_diff1 = vector1 / np.linalg.norm(vector1)
    p1 = point1 + xi[0] * vector_diff1 / miles_convert
    # Shift point according to "cross" error
    ortho_vector = np.array([vector[1], -vector[0]]) * miles_convert  
    # Calculate the orthogonal vector
    vector_diff2 = ortho_vector / np.linalg.norm(ortho_vector)
    final_point = p1 + xi[1] * vector_diff2 / miles_convert
    return final_point.tolist()


def demandByWS(hurr_cat):
    """
    Returns the demand contribution by Hurricane category
    """
    if hurr_cat > 5:
        print("Error: hurricane category should be in between 0 to 5")
        exit(0)
    cat_lst = [0, 1, 2, 3, 4, 5]
    demand_frac_lst = [0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
    for cat in cat_lst:
        if hurr_cat <= cat:
            return demand_frac_lst[cat]


def summerize_result(data_frame):
    df = data_frame.describe().transpose()
    total = df.loc['Total']['mean']
    percent_lst = [c / total for c in df['mean']]
    percent_lst = list(map(lambda x: round(x, 2), percent_lst))
    df['percent'] = percent_lst
    df = df.round(2)
    df = df.rename_axis('cost')
    df = df.drop(['count'], axis=1)
    return df


def get_closest_s(data, oos, eval_type):
    """Return a list of closest in-samples to given oos
    oos : int
        index of OOS
    eval_type : str (option)
        'oos' or 'mc_tree' from args
    """
    if eval_type == 'oos':
        in_sample = data.demand_in_sample
    else:
        in_sample = {s: {} for s in range(data.S)}
        for s in range(data.S):
            samples = data.in_sample_from_tree_2ssp[s]
            for t, state in enumerate(samples):
                demand = data.demand_mssp[t][state]
                in_sample[s][t] = demand
    closest_list = [0]
    for t in range(1, data.ts_oos[oos] + 1):
        dist = 10**10
        for s in range(data.S):
            if data.ts_in_sample[s] < t:
                continue
            dist_s = 0.0
            # print('\n\n', data.demand_oos[oos], in_sample[s], '\n\n')
            for t_ in range(1, t + 1):
                for i in range(data.I):
                    dist_s += abs(data.demand_oos[oos][t_][i] -
                                  in_sample[s][t_][i])
            if dist_s < dist:
                dist = dist_s
                closest_s = s
        closest_list.append(closest_s)
    return closest_list


def compute_total_demand(data, samples):
    n_scenarios = len(samples.keys())
    for s in range(n_scenarios):
        if 0 not in samples[s].keys():
                samples[s][0] = [0.0] * data.I
    demand = copy.deepcopy(samples)
    remaining = copy.deepcopy(samples)
    for s in range(n_scenarios):
        ts = max(samples[s].keys())
        for t in range(ts + 1):
            for i in range(data.I):
                if t == 0:
                    remaining[s][t][i] = data.DP_POP[i]
                else:
                    remaining[s][t][i] = \
                        remaining[s][t-1][i] - demand[s][t-1][i]
                demand[s][t][i] = \
                    remaining[s][t][i] * samples[s][t][i]
    return demand
