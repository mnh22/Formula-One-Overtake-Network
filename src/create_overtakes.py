import pandas as pd


from load_data import load_lap_times, load_races, load_drivers, load_results, load_sprint_results

# load lap times and sort by race, driver, and then lap
lap_times = load_lap_times()
lap_times = lap_times.sort_values(by=['raceId','driverId','lap'])

# load race and driver details to connect to overtake instances later
races = load_races()
drivers = load_drivers()
results = load_results()
sprint_results = load_sprint_results()



# establish dictionaries for points
# Grand Prix points (positions 1-10; outside top 10 = 0)
GP_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Sprint Race points (positions 1-8; outside top 8 = 0)
SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


# establish which drivers should not be counted for races that resulted in disqualification
# the three status descriptions that represent disqualifications are Disqualified, Underweight, and Excluded
DSQ_status = [2, 92, 96]


# identify sprint vs regular race id values so we can apply the correct points table per overtake
sprint_race_ids = set(sprint_results['raceId'].unique())

races_2025 = races[races['year'] == 2025]
race_ids_2025 = set(races_2025['raceId'].unique())

lap_times = lap_times[lap_times['raceId'].isin(race_ids_2025)].copy()
results = results[results['raceId'].isin(race_ids_2025)].copy()
sprint_results = sprint_results[sprint_results['raceId'].isin(race_ids_2025)].copy()

print(f"2025 race IDs found: {len(race_ids_2025)}")
print(f"  GP races:     {len(race_ids_2025 - sprint_race_ids)}")
print(f"  Sprint races: {len(race_ids_2025 & sprint_race_ids)}")


# remove rows for driver/race instances that resulted in a DSQ status (defined above)
dsq_drivers = results[results['statusId'].isin(DSQ_status)][['raceId', 'driverId']]

dsq_drivers_sprint = sprint_results[sprint_results['statusId'].isin(DSQ_status)][['raceId', 'driverId']]

dsq_drivers = pd.concat([dsq_drivers, dsq_drivers_sprint], ignore_index=True)

lap_times = lap_times.merge(
    dsq_drivers.assign(remove_flag=True),
    on=['raceId', 'driverId'],
    how='left'
)

# track how many removed rows there are for awareness and to sanity check later
removed_rows = lap_times['remove_flag'].sum()
lap_times = lap_times[lap_times['remove_flag'].isna()].drop(columns=['remove_flag'])
print(f"Removed {removed_rows} lap entries from DSQ drivers.")

# calculate number of races per driver
races_per_driver = results.groupby('driverId')['raceId'].nunique().reset_index()
races_per_driver.columns = ['driverId', 'race_count']

# add sprint races to the count (they are saved in a separate CSV file)
sprint_races_per_driver = sprint_results.groupby('driverId')['raceId'].nunique().reset_index()
sprint_races_per_driver.columns = ['driverId', 'sprint_count']

# merge and sum
driver_race_totals = races_per_driver.merge(
    sprint_races_per_driver, 
    on='driverId', 
    how='outer'
).fillna(0)
driver_race_totals['total_races'] = driver_race_totals['race_count'] + driver_race_totals['sprint_count']

# define minimum race threshold (10 is about a half of the season)
MIN_RACES = 10

# determine the list of drivers who meet the threshold
qualified_drivers = driver_race_totals[driver_race_totals['total_races'] >= MIN_RACES]['driverId'].tolist()

print(f"\nDriver filtering:")
print(f"Total unique drivers: {len(driver_race_totals)}")
print(f"Drivers with >= {MIN_RACES} races: {len(qualified_drivers)}")
print(f"Drivers filtered out: {len(driver_race_totals) - len(qualified_drivers)}")

# filter lap_times to only include qualified drivers
lap_times_before = len(lap_times)
lap_times = lap_times[lap_times['driverId'].isin(qualified_drivers)]
lap_times_after = len(lap_times)
print(f"Lap times filtered: {lap_times_before} -> {lap_times_after} ({lap_times_before - lap_times_after} removed)")

# save cleaned and filtered lap times
lap_times.to_csv(r"data\\analysis_datasets\\lap_times_cleaned.csv", index=False)
print(f"Saved cleaned lap times dataset with drivers having >= {MIN_RACES} races.")


# figure out previous position from prior lap to determine if position changed compared to current position
lap_times = lap_times.dropna(subset=['position'])
lap_times['prev_position'] = lap_times.groupby(['raceId','driverId'])['position'].shift(1)
lap_times['overtake'] = lap_times['prev_position'] - lap_times['position']

# overtakes are symmetric so keep positive values only W.L.O.G.
overtakes_gained = lap_times[lap_times['overtake']>0].copy()

# record every overtake interaction individually and include drivers who were involved in each overtake
lap_lookup = lap_times.set_index(['raceId', 'lap', 'prev_position'])['driverId'].to_dict()
rows = []
for _, row in overtakes_gained.iterrows():
    overtaker = int(row['driverId'])
    prev_pos = int(row['prev_position'])
    race_id = int(row['raceId'])
    lap = int(row['lap'])
    net_gain = int(row['overtake'])
    
    # apply correct points to overtakes base on positions and race type
    pts_table = SPRINT_POINTS if race_id in sprint_race_ids else GP_POINTS

    for i in range(1, net_gain + 1):
        overtaken_pos = prev_pos - i  # position of the driver being passed

        overtaken = lap_lookup.get((race_id, lap, overtaken_pos))
        if overtaken is None:
            continue

        # don't count self-overtakes
        if overtaken == overtaker:

            continue

        # position of the overtaker before and after this specific (singular) pass
        overtaker_pos_before = prev_pos - (i - 1)
        overtaker_pos_after  = prev_pos - i

        # point value is the improvement in championship points for this one position gain
        pts_gained = pts_table.get(overtaker_pos_after, 0) - pts_table.get(overtaker_pos_before, 0)


        rows.append({
            'raceId':               race_id,
            'lap':                  lap,
            'overtakerId':          overtaker,
            'overtakenId':          overtaken,
            'net_positions_gained': 1,
            'point_value':          pts_gained,
            'is_sprint':            race_id in sprint_race_ids,
            'overtaker_prev_pos':   overtaker_pos_before,
            'overtaker_curr_pos':   overtaker_pos_after,
            'overtaken_prev_pos':   overtaken_pos,
            'overtaken_curr_pos':   overtaken_pos + 1
        })

overtakes_gained = pd.DataFrame(rows).copy()




races_subset = races_2025[['raceId', 'year', 'round', 'name']]

overtakes_gained = overtakes_gained.merge(races_subset, on='raceId', how='left')



# prep driver information to add for easy review later on

drivers_copy = drivers.copy()

drivers_copy['full_name'] = drivers_copy['forename'] + ' ' + drivers_copy['surname']



# merge overtaker and overtaken info



overtakes_gained = overtakes_gained.merge(

    drivers_copy[['driverId', 'full_name']], 

    left_on='overtakerId', 

    right_on='driverId', 

    how='left')

overtakes_gained.rename(columns={'full_name': 'overtaker_name'}, inplace=True)

overtakes_gained.drop(columns=['driverId'], inplace=True)



overtakes_gained = overtakes_gained.merge(

    drivers_copy[['driverId', 'full_name']], 

    left_on='overtakenId', 

    right_on='driverId', 

    how='left')

overtakes_gained.rename(columns={'full_name': 'overtaken_name'}, inplace=True)

overtakes_gained.drop(columns=['driverId'], inplace=True)








# reorder columns for ease of use later
overtakes_gained = overtakes_gained[
    [
        'raceId', 'year', 'round', 'name', 'lap', 'is_sprint',
        'overtakerId', 'overtaker_name', 'overtakenId', 'overtaken_name',
        'net_positions_gained', 'point_value',
        'overtaker_prev_pos', 'overtaker_curr_pos',
        'overtaken_prev_pos', 'overtaken_curr_pos'
    ]
]

# ensure final csv shows sequential order of overtakes using lap as base time step
overtakes_gained = overtakes_gained.sort_values(['round','lap'])
overtakes_gained['global_lap'] = range(1, len(overtakes_gained) + 1)

# save overtake dataset
overtakes_gained.to_csv(r'data\\analysis_datasets\\overtakes.csv', index=False)

print(f"\nOvertakes dataset created with {len(overtakes_gained)} rows (one per overtaken driver).")
print(f"All overtakes involve drivers with >= {MIN_RACES} races.")
print(overtakes_gained.head())
print(overtakes_gained.info())
print(f"Total overtakes recorded: {len(overtakes_gained)}")

