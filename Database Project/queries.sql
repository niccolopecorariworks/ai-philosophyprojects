-- ============================================================
-- SQL Queries – US Public and Private Schools
-- Database course project (Sapienza University of Rome)
-- Schema: pubschools, prischools, collegeuni
-- ============================================================

-- Query 1: Public high schools located in a city that also has a college
SELECT pubschools.NAME AS pubhighschools, collegeuni.NAME AS college, collegeuni.CITY AS city
FROM pubschools
JOIN collegeuni ON pubschools.LEVEL_ = 'HIGH' AND pubschools.CITY = collegeuni.CITY
ORDER BY CITY;


-- Query 2: Number of public and private schools per state
SELECT COUNT(DISTINCT pubschools.NAME) AS public, COUNT(DISTINCT prischools.NAME) AS private, pubschools.STATE AS state
FROM pubschools
JOIN prischools ON pubschools.STATE = prischools.STATE
GROUP BY STATE;


-- Query 3: Colleges where the student-to-dormitory ratio is >= 2, and the inverse case

-- 3a: Ratio of students to dormitory capacity >= 2
SELECT collegeuni.NAME AS college, collegeuni.STATE AS state, collegeuni.TOT_ENROLL AS numstudenti, collegeuni.DORM_CAP
FROM collegeuni
WHERE collegeuni.TOT_ENROLL / collegeuni.DORM_CAP >= 2;

-- 3b: Ratio of dormitory capacity to students >= 2
SELECT collegeuni.NAME AS college, collegeuni.STATE AS state, collegeuni.TOT_ENROLL AS numstudenti, collegeuni.DORM_CAP
FROM collegeuni
WHERE collegeuni.DORM_CAP / collegeuni.TOT_ENROLL >= 2;


-- Query 4: Total number of public and private school students in Alaska
SELECT SUM(pubschools.ENROLLMENT) AS pubstudents, SUM(prischools.ENROLLMENT) AS pristudents, pubschools.STATE
FROM pubschools
JOIN prischools ON pubschools.ENROLLMENT > 0 AND prischools.ENROLLMENT > 0
    AND pubschools.STATE = 'AK' AND prischools.STATE = 'AK'
GROUP BY pubschools.STATE;


-- Query 5: Student-to-teacher ratio for public schools in Wisconsin
SELECT DISTINCT pubschools.NAME AS pubschool, pubschools.ENROLLMENT AS totpubstud, pubschools.FT_TEACHER AS pubteachers,
    pubschools.ENROLLMENT / pubschools.FT_TEACHER AS pubteacherrate
FROM pubschools
WHERE pubschools.STATE = 'WI' AND pubschools.ENROLLMENT > 1 AND pubschools.FT_TEACHER > 1;


-- Query 6: Public schools and, where present, private schools by county in a state
SELECT pubschools.NAME AS pubschool, prischools.NAME AS prischool, pubschools.COUNTY AS county
FROM pubschools
LEFT JOIN prischools ON pubschools.STATE = 'ND' AND prischools.STATE = 'ND'
ORDER BY county;


-- Query 7: Colleges and, where present, private/public schools by county in the same state

-- 7a: Colleges with private schools in the same county
SELECT prischools.NAME AS prischool, collegeuni.NAME AS college, collegeuni.COUNTY
FROM prischools
RIGHT JOIN collegeuni ON prischools.STATE = 'ND' AND collegeuni.STATE = 'ND'
ORDER BY collegeuni.COUNTY;

-- 7b: Colleges with public schools in the same county
SELECT pubschools.NAME AS pubschool, collegeuni.NAME AS college, collegeuni.COUNTY
FROM pubschools
RIGHT JOIN collegeuni ON pubschools.STATE = 'ND' AND collegeuni.STATE = 'ND'
ORDER BY collegeuni.COUNTY;


-- Query 8: Schools with the highest enrollment in a state (public, private, and college level)

-- 8a: Single state, using nested subqueries and UNION across school types
SELECT pubschools.NAME AS school, pubschools.STATE AS state, pubschools.ENROLLMENT AS numstu
FROM pubschools
WHERE pubschools.STATE = 'ND'
    AND pubschools.ENROLLMENT = (
        SELECT MAX(pubschools.ENROLLMENT) FROM pubschools WHERE pubschools.STATE = 'ND'
    )
UNION
SELECT prischools.NAME AS school, prischools.STATE AS state, prischools.ENROLLMENT AS numstu
FROM prischools
WHERE prischools.STATE = 'ND'
    AND prischools.ENROLLMENT = (
        SELECT MAX(prischools.ENROLLMENT) FROM prischools WHERE prischools.STATE = 'ND'
    )
UNION
SELECT collegeuni.NAME AS school, collegeuni.STATE AS state, collegeuni.TOT_ENROLL AS numstu
FROM collegeuni
WHERE collegeuni.STATE = 'ND'
    AND collegeuni.TOT_ENROLL = (
        SELECT MAX(collegeuni.TOT_ENROLL) FROM collegeuni WHERE collegeuni.STATE = 'ND'
    );

-- 8b: All states at once, joining top public, private and college enrollment per state
SELECT pubschools.NAME AS pubschool, pubschools.ENROLLMENT AS numstupub, prischools.NAME AS prischool, prischools.ENROLLMENT AS numstupri,
    collegeuni.NAME AS college, collegeuni.TOT_ENROLL AS numstucoll, collegeuni.STATE AS state
FROM pubschools
JOIN prischools ON pubschools.STATE = prischools.STATE
    AND pubschools.ENROLLMENT = (
        SELECT MAX(pubschools.ENROLLMENT) FROM pubschools WHERE pubschools.STATE = prischools.STATE
    )
    AND prischools.ENROLLMENT = (
        SELECT MAX(prischools.ENROLLMENT) FROM prischools WHERE prischools.STATE = pubschools.STATE
    )
JOIN collegeuni ON collegeuni.STATE = prischools.STATE
    AND collegeuni.TOT_ENROLL = (
        SELECT MAX(collegeuni.TOT_ENROLL) FROM collegeuni WHERE collegeuni.STATE = prischools.STATE
    )
ORDER BY collegeuni.STATE;


-- Query 9: Total number of colleges per state and average sector value
SELECT collegeuni.STATE AS stato, COUNT(collegeuni.NAME) AS collegecount, AVG(collegeuni.SECTOR) AS avgsector
FROM collegeuni
WHERE collegeuni.SECTOR > 0 AND collegeuni.SECTOR < 99
GROUP BY collegeuni.STATE
ORDER BY avgsector DESC;


-- Query 10: Minimum number of students and teachers per state, public and private
SELECT pubschools.STATE AS state, MIN(pubschools.ENROLLMENT) AS minpub, MIN(prischools.ENROLLMENT) AS minpri,
    MIN(pubschools.FT_TEACHER) AS minteachpub, MIN(prischools.FT_TEACHER) AS minteachpri
FROM pubschools
JOIN prischools ON pubschools.STATE = prischools.STATE
    AND pubschools.ENROLLMENT > 0 AND prischools.ENROLLMENT > 0
    AND pubschools.FT_TEACHER > 0 AND prischools.FT_TEACHER > 0
GROUP BY pubschools.STATE;
