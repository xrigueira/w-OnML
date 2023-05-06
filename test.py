import tictoc

labels = [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
predicted_labels = [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0]

def find_anomalies(labels):
    anomalies = []
    i = 0
    while i < len(labels):
        if labels[i] == 1:
            j = i + 1
            while j < len(labels) and labels[j] == 1:
                j += 1
            if j < len(labels) and labels[j] == 0:
                anomalies.append((i, j))
            i = j
        else:
            i += 1
    return anomalies

def match_percentage(labels, predicted_labels, anomaly_tail):
    anomalies = find_anomalies(labels)
    total_matches = 0
    total_items = 0
    for start, end in anomalies:
        anomaly_length = end - start
        segment = predicted_labels[start:end+int(anomaly_length*anomaly_tail)]
        print(labels[start:end+int(anomaly_length*anomaly_tail)])
        print(segment)
        total_matches += sum(a == b for a, b in zip(labels[start:end+int(anomaly_length*anomaly_tail)], segment))
        print('Total matches',total_matches)
        total_items += (anomaly_length + int(anomaly_length*anomaly_tail))
        print('Total items', total_items)
    return total_matches / total_items if total_items > 0 else 1.0

result = match_percentage(labels=labels, predicted_labels=predicted_labels, anomaly_tail=1)

print(result)

# from combinatorics import Metric

# metric = Metric(labels=labels, predicted_labels=predicted_labels)
# result = metric.match_percentage()

# print(result)