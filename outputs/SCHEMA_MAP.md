# AGWA live schema map

- Server: `10.4.33-MariaDB`  |  DB: `agentinfo`  |  User: `agentinfo@%`

## Tables (information_schema estimates)

| table_name            | engine   |       table_rows |   data_gb |   index_gb |
|:----------------------|:---------|-----------------:|----------:|-----------:|
| hits                  | TokuDB   |      6.05295e+09 |    314.85 |     839.4  |
| agentgraph            | TokuDB   |      1.56034e+08 |     18.57 |      10.28 |
| agentgraph2           | TokuDB   |      1.28055e+08 |      1.91 |       3.7  |
| url                   | TokuDB   |      7.12046e+07 |      3.11 |       4.62 |
| ip                    | TokuDB   |      5.21282e+07 |      1.87 |       2.51 |
| exturl                | InnoDB   |      3.67168e+07 |      3.47 |       4.67 |
| hackip                | InnoDB   |      2.97438e+06 |      0.17 |       0.04 |
| agent                 | TokuDB   |      2.27412e+06 |      0.72 |       0.32 |
| pearsonall            | InnoDB   |      1.95052e+06 |      0.12 |       0.03 |
| pearsonbrowser        | InnoDB   |      1.39364e+06 |      0.08 |       0.02 |
| tmpallexcel           | InnoDB   | 568634           |      0.02 |       0.04 |
| tmpbrowserexcel       | InnoDB   | 488259           |      0.02 |       0.02 |
| ip2activeip           | InnoDB   | 199515           |      0.01 |       0.01 |
| url2agent_honeypot    | InnoDB   | 194639           |      0.01 |       0    |
| activeip              | InnoDB   | 101058           |      0.01 |       0    |
| tmpbotexcel           | InnoDB   |  78921           |      0    |       0    |
| pearsonbot            | TokuDB   |  73260           |      0    |       0    |
| domain                | TokuDB   |   8780           |      0    |       0    |
| pearson               | InnoDB   |   8077           |      0    |       0    |
| tmp4                  | InnoDB   |   2289           |      0    |       0    |
| tmp2                  | InnoDB   |   1952           |      0    |       0    |
| tmp3                  | InnoDB   |   1216           |      0    |       0    |
| whiteip               | InnoDB   |    279           |      0    |       0    |
| tmp4b                 | InnoDB   |    200           |      0    |       0    |
| emptydays             | InnoDB   |    110           |      0    |       0    |
| emailip               | InnoDB   |     89           |      0    |       0    |
| headip                | InnoDB   |     39           |      0    |       0    |
| exturl2agent_honeypot | InnoDB   |      0           |      0    |       0    |
| tmpexcel              | InnoDB   |      0           |      0    |       0    |
| hackip_w              |          |    nan           |    nan    |     nan    |

## `hits` columns

| Field    | Type       | Null   | Key   | Default             | Extra                         |
|:---------|:-----------|:-------|:------|:--------------------|:------------------------------|
| h_id     | bigint(20) | NO     | MUL   |                     | auto_increment                |
| h_a_id   | int(11)    | YES    | MUL   |                     |                               |
| h_ccode  | varchar(8) | YES    |       |                     |                               |
| h_prox   | tinyint(4) | YES    |       |                     |                               |
| h_mode   | varchar(8) | YES    | MUL   |                     |                               |
| h_i_id   | bigint(20) | YES    |       |                     |                               |
| h_d_id   | int(11)    | YES    |       |                     |                               |
| h_u_id   | bigint(20) | YES    | MUL   |                     |                               |
| h_e_id   | bigint(20) | YES    | MUL   |                     |                               |
| h_status | int(11)    | YES    |       |                     |                               |
| h_ts     | timestamp  | NO     |       | current_timestamp() | on update current_timestamp() |

### `hits` indexes

| Key_name   |   Seq_in_index | Column_name   |   Non_unique |
|:-----------|---------------:|:--------------|-------------:|
| h_id       |              1 | h_id          |            1 |
| h_1        |              1 | h_a_id        |            1 |
| h_1        |              2 | h_ts          |            1 |
| h_2        |              1 | h_u_id        |            1 |
| h_2        |              2 | h_status      |            1 |
| h_2        |              3 | h_ts          |            1 |
| h_3        |              1 | h_mode        |            1 |
| h_3        |              2 | h_status      |            1 |
| h_3        |              3 | h_ts          |            1 |
| h_4        |              1 | h_e_id        |            1 |
| h_4        |              2 | h_status      |            1 |
| h_4        |              3 | h_ts          |            1 |
| h_5        |              1 | h_a_id        |            1 |
| h_5        |              2 | h_status      |            1 |
| h_6        |              1 | h_u_id        |            1 |
| h_6        |              2 | h_a_id        |            1 |
| h_6        |              3 | h_ts          |            1 |

## `agent` columns

| Field              | Type         | Null   | Key   | Default   | Extra          |
|:-------------------|:-------------|:-------|:------|:----------|:---------------|
| a_id               | bigint(20)   | NO     | MUL   |           | auto_increment |
| a_name             | varchar(255) | YES    | UNI   |           |                |
| a_start            | date         | YES    |       |           |                |
| a_last             | date         | YES    |       |           |                |
| a_valuedays        | int(11)      | YES    |       |           |                |
| a_realvaluedays    | int(11)      | YES    |       |           |                |
| a_mediumvaluedays  | int(11)      | YES    |       |           |                |
| a_emediumvaluedays | int(11)      | YES    |       |           |                |
| a_totalhit         | int(11)      | YES    |       |           |                |
| a_stat_404         | int(11)      | YES    |       |           |                |
| a_stat_delete      | int(11)      | YES    |       |           |                |
| a_stat_get         | int(11)      | YES    |       |           |                |
| a_stat_head        | int(11)      | YES    |       |           |                |
| a_stat_post        | int(11)      | YES    |       |           |                |
| a_stat_put         | int(11)      | YES    |       |           |                |
| a_stat_wplogin     | int(11)      | YES    |       |           |                |
| a_stat_xmlrpc      | int(11)      | YES    |       |           |                |
| a_start_5p         | date         | YES    |       |           |                |
| a_last_5p          | date         | YES    |       |           |                |
| a_cpt_10           | int(11)      | YES    |       |           |                |
| a_cpt_20           | int(11)      | YES    |       |           |                |
| a_cpt_30           | int(11)      | YES    |       |           |                |
| a_cpt_40           | int(11)      | YES    |       |           |                |
| a_cpt_50           | int(11)      | YES    |       |           |                |
| a_cpt_60           | int(11)      | YES    |       |           |                |
| a_cpt_70           | int(11)      | YES    |       |           |                |
| a_cpt_80           | int(11)      | YES    |       |           |                |
| a_cpt_90           | int(11)      | YES    |       |           |                |
| a_cpt_100          | int(11)      | YES    |       |           |                |
| a_date_10          | date         | YES    |       |           |                |
| a_date_20          | date         | YES    |       |           |                |
| a_date_30          | date         | YES    |       |           |                |
| a_date_40          | date         | YES    |       |           |                |
| a_date_50          | date         | YES    |       |           |                |
| a_date_60          | date         | YES    |       |           |                |
| a_date_70          | date         | YES    |       |           |                |
| a_date_80          | date         | YES    |       |           |                |
| a_date_90          | date         | YES    |       |           |                |
| a_cpt_5            | int(11)      | YES    |       |           |                |
| a_cpt_15           | int(11)      | YES    |       |           |                |
| a_cpt_25           | int(11)      | YES    |       |           |                |
| a_cpt_35           | int(11)      | YES    |       |           |                |
| a_cpt_45           | int(11)      | YES    |       |           |                |
| a_cpt_55           | int(11)      | YES    |       |           |                |
| a_cpt_65           | int(11)      | YES    |       |           |                |
| a_cpt_75           | int(11)      | YES    |       |           |                |
| a_cpt_85           | int(11)      | YES    |       |           |                |
| a_cpt_95           | int(11)      | YES    |       |           |                |
| a_date_15          | date         | YES    |       |           |                |
| a_date_25          | date         | YES    |       |           |                |
| a_date_35          | date         | YES    |       |           |                |
| a_date_45          | date         | YES    |       |           |                |
| a_date_55          | date         | YES    |       |           |                |
| a_date_65          | date         | YES    |       |           |                |
| a_date_75          | date         | YES    |       |           |                |
| a_date_85          | date         | YES    |       |           |                |
| a_date_95          | date         | YES    |       |           |                |

### `agent` indexes

| Key_name   |   Seq_in_index | Column_name   |   Non_unique |
|:-----------|---------------:|:--------------|-------------:|
| a_1        |              1 | a_name        |            0 |
| a_id       |              1 | a_id          |            1 |

## `domain` columns

| Field   | Type         | Null   | Key   | Default   | Extra          |
|:--------|:-------------|:-------|:------|:----------|:---------------|
| d_id    | int(11)      | NO     | MUL   |           | auto_increment |
| d_name  | varchar(100) | YES    | UNI   |           |                |

### `domain` indexes

| Key_name   |   Seq_in_index | Column_name   |   Non_unique |
|:-----------|---------------:|:--------------|-------------:|
| d_1        |              1 | d_name        |            0 |
| d_id       |              1 | d_id          |            1 |

## `url` columns

| Field       | Type         | Null   | Key   | Default   | Extra          |
|:------------|:-------------|:-------|:------|:----------|:---------------|
| u_id        | bigint(20)   | NO     | MUL   |           | auto_increment |
| u_name      | varchar(255) | YES    | UNI   |           |                |
| u_extension | varchar(16)  | YES    |       |           |                |
| u_honeypot  | int(11)      | YES    | MUL   |           |                |

### `url` indexes

| Key_name   |   Seq_in_index | Column_name   |   Non_unique |
|:-----------|---------------:|:--------------|-------------:|
| u_1        |              1 | u_name        |            0 |
| u_id       |              1 | u_id          |            1 |
| u_2        |              1 | u_honeypot    |            1 |

## `ip` columns

| Field   | Type        | Null   | Key   |   Default | Extra          |
|:--------|:------------|:-------|:------|----------:|:---------------|
| i_id    | bigint(20)  | NO     | MUL   |           | auto_increment |
| i_name  | varchar(50) | YES    | MUL   |           |                |
| i_agent | int(11)     | YES    |       |           |                |
| i_year  | smallint(6) | YES    |       |           |                |
| i_month | tinyint(4)  | YES    |       |           |                |
| i_count | int(11)     | YES    |       |         0 |                |
| i_ccode | varchar(8)  | YES    |       |           |                |
| i_proxy | tinyint(4)  | YES    |       |         0 |                |

### `ip` indexes

| Key_name   |   Seq_in_index | Column_name   |   Non_unique |
|:-----------|---------------:|:--------------|-------------:|
| i_1        |              1 | i_name        |            0 |
| i_1        |              2 | i_agent       |            0 |
| i_1        |              3 | i_year        |            0 |
| i_1        |              4 | i_month       |            0 |
| i_id       |              1 | i_id          |            1 |

## Probes

### hits date range (h_ts)

_(probe failed or timed out: OperationalError: (1030, 'Got error 1152 "Unknown error 1152" from storage engine TokuDB'))_

### distinct HTTP status codes (confirm 403/429 exist)

_(probe failed or timed out: OperationalError: (1969, 'Query execution was interrupted (max_statement_time exceeded)'))_

### robots.txt url rows (u_last10 ending in robots.txt)

_(probe failed or timed out: DatabaseError: Execution failed on sql 'SELECT u_id, u_last10 FROM url WHERE u_last10 LIKE '%robots.txt' LIMIT 5': (1054, "Unknown colu)_

### request methods (h_mode)

_(probe failed or timed out: OperationalError: (1030, 'Got error 1152 "Unknown error 1152" from storage engine TokuDB'))_
