def concordance_index(event_times, predicted_times, event_observed):
    def valid_comparison(time_a, time_b, event_a, event_b):
        if time_a == time_b:
            return event_a != event_b
        elif event_a and event_b:
            return True
        elif event_a and time_a < time_b:
            return True
        elif event_b and time_b < time_a:
            return True
        else:
            return False

    def concordance_value(time_a, time_b, pred_a, pred_b):
        if pred_a == pred_b:
            return 0.5
        elif pred_a < pred_b:
            return (time_a < time_b) or (time_a == time_b and event_a and not event_b)
        else:
            return (time_a > time_b) or (time_a == time_b and not event_a and event_b)

    paircount = 0.0
    csum = 0.0

    for a in range(0,len(event_times)):
        time_a = event_times[a]
        pred_a = predicted_times[a]
        event_a = event_observed[a]
        for b in range(a + 1, len(event_times)):
            time_b = event_times[b]
            pred_b = predicted_times[b]
            event_b = event_observed[b]

            if valid_comparison(time_a, time_b, event_a, event_b):
                paircount += 1.0
                csum += concordance_value(time_a ,time_b, pred_a, pred_b)

    if paircount == 0:
        raise ZeroDivisionError("No admissable pairs in the dataset.")
    return csum / paircount
