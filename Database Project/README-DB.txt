# Database Queries – US Public and Private Schools - N. Pecorari, G. Gatto

## Overview
A set of SQL queries developed for the Database course, working on an existing 
relational schema covering public and private educational institutions in the 
US, from kindergarten to college level. The project focuses on extracting and 
aggregating data through complex queries rather than schema design.

## What's inside
- Queries using JOIN operations across multiple tables
- Aggregation queries (GROUP BY, COUNT, and other aggregate functions)
- Subqueries for nested data retrieval

## Example queries

**Schools with the highest enrollment in a state (public, private, and college level)**
Uses nested subqueries to find the maximum enrollment per school type, combined 
with UNION across the three categories.

**Student-to-teacher ratio in a given state**
Calculates the ratio using a computed field directly in the SELECT statement.

**Number of public and private schools per county in a state**
Uses LEFT JOIN to include public schools even when no matching private school 
exists in the same county.

**Total number of colleges per state with average sector value**
Aggregates data with COUNT and AVG, grouped and ordered by state.

## Tech stack
MySQL / SQL

## Files
- `queries.sql` — full set of queries with comments explaining each one

## Notes
Developed as a group assignment for the "Base di Dati e Rappresentazione della 
Conoscenza" course at Sapienza University of Rome.