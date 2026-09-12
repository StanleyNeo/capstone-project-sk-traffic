-- =====================================================================
-- Part 1 - Task 1: SQL-Based Traffic Analysis
-- Dataset: Metro Interstate Traffic Volume (I-94 westbound, MN)
-- Database: data/traffic.db  (table: traffic)
-- Run with: python part1_data_analytics/sql/run_analysis.py
-- =====================================================================

-- @name: 1.1a Row count (verification)
SELECT COUNT(*) AS rows_loaded FROM traffic;

-- @name: 1.1b Date range (verification)
SELECT MIN(date_time) AS first_timestamp, MAX(date_time) AS last_timestamp FROM traffic;

-- @name: 1.1c Sample rows (verification)
SELECT * FROM traffic ORDER BY date_time LIMIT 5;

-- @name: 1.1d Duplicate timestamp check (rows vs distinct timestamps)
SELECT COUNT(*) AS total_rows,
       COUNT(DISTINCT date_time) AS distinct_timestamps,
       COUNT(*) - COUNT(DISTINCT date_time) AS duplicate_timestamps
FROM traffic;

-- @name: 1.2a Total yearly traffic volume 2012-2017 (raw totals)
SELECT substr(date_time, 1, 4) AS year,
       COUNT(*) AS hours_recorded,
       SUM(traffic_volume) AS total_volume
FROM traffic
WHERE substr(date_time, 1, 4) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- @name: 1.2b Year-on-year change in total volume
WITH yearly AS (
    SELECT substr(date_time, 1, 4) AS year,
           SUM(traffic_volume) AS total_volume
    FROM traffic
    WHERE substr(date_time, 1, 4) BETWEEN '2012' AND '2017'
    GROUP BY year
)
SELECT year,
       total_volume,
       total_volume - LAG(total_volume) OVER (ORDER BY year) AS yoy_change,
       ROUND(100.0 * (total_volume - LAG(total_volume) OVER (ORDER BY year))
             / LAG(total_volume) OVER (ORDER BY year), 1) AS yoy_pct_change
FROM yearly
ORDER BY year;

-- @name: 1.2c Normalised view: average HOURLY volume per year (corrects for partial years)
SELECT substr(date_time, 1, 4) AS year,
       COUNT(*) AS hours_recorded,
       ROUND(AVG(traffic_volume), 1) AS avg_hourly_volume
FROM traffic
WHERE substr(date_time, 1, 4) BETWEEN '2012' AND '2017'
GROUP BY year
ORDER BY year;

-- @name: 1.3a Holiday flag quirk: 'None' is a literal STRING, and only the 00:00 hour is tagged
-- NOTE: the CSV stores the text 'None' (not a NULL) for non-holidays,
-- and the flag appears only on the 00:00 row of each holiday.
-- Part 2 cleaning will standardise 'None' to NULL.
SELECT holiday,
       COUNT(*) AS flagged_hours,
       COUNT(DISTINCT substr(date_time, 1, 10)) AS distinct_days
FROM traffic
WHERE holiday <> 'None'
GROUP BY holiday
ORDER BY holiday;

-- @name: 1.3b Temperature on New Year's Day and Labor Day 2015-2017 (matched by DATE)
WITH holiday_dates(holiday_name, holiday_date) AS (
    VALUES
        ('New Years Day', '2015-01-01'),
        ('New Years Day', '2016-01-01'),
        ('New Years Day', '2017-01-01'),
        ('Labor Day',     '2015-09-07'),
        ('Labor Day',     '2016-09-05'),
        ('Labor Day',     '2017-09-04')
)
SELECT h.holiday_name,
       h.holiday_date,
       COUNT(t.date_time) AS hours_recorded,
       ROUND(AVG(t.temp - 273.15), 1) AS avg_temp_c,
       ROUND(MIN(t.temp - 273.15), 1) AS min_temp_c,
       ROUND(MAX(t.temp - 273.15), 1) AS max_temp_c,
       SUM(t.traffic_volume) AS total_traffic
FROM holiday_dates h
LEFT JOIN traffic t ON substr(t.date_time, 1, 10) = h.holiday_date
GROUP BY h.holiday_name, h.holiday_date
ORDER BY h.holiday_name, h.holiday_date;

-- @name: 1.3c Context: average holiday vs non-holiday daily traffic (all years)
WITH daily AS (
    SELECT substr(date_time, 1, 10) AS day,
           SUM(traffic_volume) AS daily_volume
    FROM traffic
    GROUP BY day
),
flagged AS (
    SELECT DISTINCT substr(date_time, 1, 10) AS holiday_day
    FROM traffic
    WHERE holiday <> 'None'
)
SELECT CASE WHEN f.holiday_day IS NOT NULL THEN 'Holiday' ELSE 'Non-holiday' END AS day_type,
       COUNT(*) AS days,
       ROUND(AVG(d.daily_volume), 0) AS avg_daily_volume
FROM daily d
LEFT JOIN flagged f ON d.day = f.holiday_day
GROUP BY day_type;