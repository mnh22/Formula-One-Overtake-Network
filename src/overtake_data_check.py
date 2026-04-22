import pandas as pd
import matplotlib.pyplot as plt

ot = pd.read_csv('data/analysis_datasets/overtakes.csv')
print(ot.head())
print(ot.info())
print(ot.describe())
print(f"\nYears present: {ot['year'].unique()}")
print(f"Is sprint breakdown:\n{ot['is_sprint'].value_counts()}")



# fill missing years so the plot doesn't skip years
all_years = range(1996, 2026)



# calculate net overtakes by year

net_overtakes_list = []

for year, year_data in ot.groupby('year'):

    pair_summary = (
        year_data.groupby(['overtakerId', 'overtakenId'])['net_positions_gained']
        .sum().reset_index()
    )

    processed_pairs = set()

    for _, row in pair_summary.iterrows():
        driver_a = row['overtakerId']
        driver_b = row['overtakenId']
        pair = tuple(sorted([driver_a, driver_b]))

        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)

        # count overtaker overtakes of overtaken
        a_to_b = pair_summary[
            (pair_summary['overtakerId'] == driver_a) &
            (pair_summary['overtakenId'] == driver_b)
        ]['net_positions_gained'].sum()

        # count overtaken overtakes of overtaker
        b_to_a = pair_summary[
            (pair_summary['overtakerId'] == driver_b) &
            (pair_summary['overtakenId'] == driver_a)
        ]['net_positions_gained'].sum()

        # calculate net overtakes
        net = abs(a_to_b - b_to_a)

        if net > 0:
            net_overtakes_list.append({
                'year': year,
                'net_overtakes': net
            })




print(f"Point value distribution (count per value):")
print(ot['point_value'].value_counts().sort_index())

# spot-check: overtakes crossing into/out of points-paying positions should have non-zero point_value

into_points_gp = ot[
    (ot['is_sprint'] == False) &
    (ot['overtaker_prev_pos'] == 11) &
    (ot['overtaker_curr_pos'] == 10)
]
print(f"\nGP overtakes from P11 -> P10 (should be point_value=1): {len(into_points_gp)} rows")
if not into_points_gp.empty:
    print(into_points_gp[['overtaker_name', 'overtaken_name', 'point_value']].head())

# P10->P9 in GP should be worth 1 point (2 - 1 = 1)
p10_to_p9 = ot[
    (ot['is_sprint'] == False) &
    (ot['overtaker_prev_pos'] == 10) &
    (ot['overtaker_curr_pos'] == 9)
]
print(f"\nGP overtakes from P10 -> P9 (should be point_value=1): {len(p10_to_p9)} rows")
if not p10_to_p9.empty:
    print(p10_to_p9[['overtaker_name', 'overtaken_name', 'point_value']].head())

# P2->P1 in GP should be worth 7 points (25 - 18 = 7)
p2_to_p1 = ot[
    (ot['is_sprint'] == False) &
    (ot['overtaker_prev_pos'] == 2) &
    (ot['overtaker_curr_pos'] == 1)
]
print(f"\nGP overtakes from P2 -> P1 (should be point_value=7): {len(p2_to_p1)} rows")
if not p2_to_p1.empty:
    print(p2_to_p1[['overtaker_name', 'overtaken_name', 'point_value']].head())

# Sprint: P8->P7 should be worth 1 point (2 - 1 = 1)
sprint_p8_to_p7 = ot[
    (ot['is_sprint'] == True) &
    (ot['overtaker_prev_pos'] == 8) &
    (ot['overtaker_curr_pos'] == 7)
]
print(f"\nSprint overtakes from P8 -> P7 (should be point_value=1): {len(sprint_p8_to_p7)} rows")
if not sprint_p8_to_p7.empty:
    print(sprint_p8_to_p7[['overtaker_name', 'overtaken_name', 'point_value']].head())

# outside points: P15->P14 should be 0 in both race types
outside_pts = ot[
    (ot['overtaker_prev_pos'] == 15) &
    (ot['overtaker_curr_pos'] == 14)
]
print(f"\nOvertakes from P15 -> P14 (should be point_value=0): {len(outside_pts)} rows")
if not outside_pts.empty:
    print(outside_pts[['overtaker_name', 'overtaken_name', 'point_value', 'is_sprint']].head())

# ── Overtakes per race ─────────────────────────────────────────────────────────
print("\n=== Overtakes per Race ===")
overtakes_by_race = ot.groupby(['round', 'name', 'is_sprint']).size().reset_index(name='overtake_count')
print(overtakes_by_race.to_string())

# ── Net overtakes plot ─────────────────────────────────────────────────────────
# aggregate total point value by race for a quick sanity-check plot
point_value_by_race = ot.groupby(['round', 'name'])['point_value'].sum().reset_index()
point_value_by_race = point_value_by_race.sort_values('round')

plt.figure(figsize=(14, 5))
plt.bar(point_value_by_race['round'], point_value_by_race['point_value'])
plt.xticks(point_value_by_race['round'], point_value_by_race['name'], rotation=90, fontsize=8)
plt.title("Total Point Value of Overtakes by Race (2025)")
plt.xlabel("Race")
plt.ylabel("Total Point Value")
plt.tight_layout()
plt.savefig("results/point_value_by_race.png")
plt.show()

# ── Drivers and unique interactions ───────────────────────────────────────────
print("\n=== Driver Coverage ===")
print(f"Unique overtakers:  {ot['overtakerId'].nunique()}")
print(f"Unique overtaken:   {ot['overtakenId'].nunique()}")
print(f"Total overtake rows: {len(ot)}")
print(f"\nRaces per round:\n{ot.groupby('round')['raceId'].nunique()}")

# ===== OTHER ANALYSES (useful for debugging mostly) =====
print("\n" + "="*50)
print("ADDITIONAL STATISTICS")
print("="*50)

overtakes_by_race = ot.groupby(['year', 'raceId']).size()
print("\nOvertakes per race (first 20):")
print(overtakes_by_race.head(20))

weak_races = overtakes_by_race[overtakes_by_race < 10]
print(f"\nRaces with <10 overtakes: {len(weak_races)}")
print(weak_races)

extra_races = overtakes_by_race[overtakes_by_race > 200]
print(f"\nRaces with >200 overtakes: {len(extra_races)}")
print(extra_races)

races_per_year = ot.groupby('year')['raceId'].nunique()
print("\nRaces per year:")
print(races_per_year)

drivers_per_year = ot.groupby('year')['overtakerId'].nunique()
print("\nUnique drivers per year:")
print(drivers_per_year)

laps_per_race = ot.groupby(['raceId']).lap.nunique()
print("\nLaps per race (20 shortest):")
print(laps_per_race.sort_values().head(10))

