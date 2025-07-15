# DOCS2025-VRPTW

Team: Route Seekers

## Experimental results

- 200-customer Instances [2025/06/19]

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Time (Ours) | Vehicles | Distance | Time | Reference | Date      | Comment                       |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------: | :------: | :------: | :--: | --------- | --------- | ----------------------------- |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |     **20**      |     2698.6      |     30      |  **20**  | 2704.57  |      | GH        | 2001      | Detailed solution by SAM::OPT |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |       20        |     2694.3      |     30      |  **18**  | 2917.89  |      | BVH       | 2001      | Detailed solution by SCR      |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |       20        |     2675.8      |     30      |  **18**  | 2707.35  |      | BSJ2      | 20-sep-07 | Detailed solution by SCR      |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |       19        |     2627.2      |     30      |  **18**  | 2643.31  |      | BSJ2      | 20-sep-07 | Detailed solution by SCR      |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |     **20**      |     2694.9      |     30      |  **20**  | 2702.05  |      | GH        | 2001      | Detailed solution by SAM::OPT |

- 800-customer Instances [2025/06/19]

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Time (Ours) | Vehicles | Distance | Time | Reference | Date                                                         |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------: | :------: | :------: | :--: | --------- | ------------------------------------------------------------ |
|                            c1_8_1                            |     **80**      |     25156.9     |     30      |  **80**  | 25030.36 |      | M         | 2002. There are questions whether the solution is valid, several authors report [80/25184.38](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_1.25184.38.sintef.txt) |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |       79        |     25069.3     |     30      |  **72**  | 26540.53 |      | CAINIAO   | Feb-19                                                       |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |       76        |     24890.5     |     30      |  **72**  | 24242.49 |      | SCR       | Oct-18                                                       |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |       73        |     24248.8     |     30      |  **72**  | 23824.17 |      | Q         | 28-oct-14                                                    |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |     **80**      |     25144.2     |     30      |  **80**  | 25166.28 |      | RP        | 25-feb-05                                                    |

Note: time in seconds.

- Improved Results: fleet minimization in priority, and then route length minimization [2025/06/28]

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Time (Ours) |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------: |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |     **20**      |   **2704.56**   |     100     |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |     **18**      |   **2917.89**   |     200     |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |     **18**      |   **2707.34**   |    1800     |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |     **18**      |     2644.59     |    1800     |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |     **20**      |   **2702.04**   |    1800     |
|                            c1_8_1                            |     **80**      |  **25184.34**   |    1800     |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |       75        |    25244.51     |    1800     |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |       73        |    24460.69     |    1800     |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |     **72**      |    24036.50     |    1800     |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |     **80**      |  **25166.23**   |    1800     |



- Improved Results: parallel computing with 8 CPU cores, larger coefficient with respect to # of vehihle  + 1 run [2025/06/29]

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Time (Ours) |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------: |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |                 |                 |             |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |                 |                 |             |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |                 |                 |             |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |     **18**      |     2643.63     |    1800     |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |                 |                 |             |
|                            c1_8_1                            |     **80**      |  **25184.34**   |    1800     |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |       74        |    25887.84     |    1800     |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |     **72**      |    24846.55     |    1800     |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |     **72**      |    24397.92     |    1800     |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |                 |                 |             |



- Improved Results: HGS parallel computing with 8 CPU cores + 10 run [2025/07/03]

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | $\bar{\text{Vehicles}}$(Ours) | $\bar{\text{Distance}}$(Ours) | Time (Ours) |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------------------------: | :---------------------------: | :---------: |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |                 |                 |                               |                               |             |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |                 |                 |                               |                               |             |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |                 |                 |                               |                               |             |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |                 |                 |                               |                               |             |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |                 |                 |                               |                               |             |
|                            c1_8_1                            |                 |                 |                               |                               |             |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |       74        |    25611.74     |       74.00 $\pm$ 0.00        |     25766.61 $\pm$ 111.68     |    1800     |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |       72        |    24846.55     |       72.00 $\pm$ 0.00        |     25002.85 $\pm$ 82.55      |    1800     |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |       72        |    24545.69     |       72.00 $\pm$ 0.00        |     24712.05 $\pm$ 139.53     |    1800     |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |                 |                 |                               |                               |             |



- Improved Results: HGS + barycenter clustering decomposition + subproblem parallel computing with 8 CPU cores + 10 run [2025/07/01]

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | $\bar{\text{Vehicles}}$(Ours) | $\bar{\text{Distance}}$(Ours) | Time (Ours) |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------------------------: | :---------------------------: | :---------: |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |                 |                 |                               |                               |             |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |                 |                 |                               |                               |             |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |                 |                 |                               |                               |             |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |                 |                 |                               |                               |             |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |                 |                 |                               |                               |             |
|                            c1_8_1                            |                 |                 |                               |                               |             |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |       74        |    25689.95     |       74.70 $\pm$ 0.48        |     25471.72 $\pm$ 276.77     |    1800     |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |       72        |    24474.38     |       72.00 $\pm$ 0.00        |     24546.45 $\pm$ 65.27      |    1800     |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |       72        |    23990.70     |       72.00 $\pm$ 0.00        |     24252.49 $\pm$ 232.85     |    1800     |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |                 |                 |                               |                               |             |

- Submission Results：[2025/07/07]

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Duration (Ours) | Time (Ours) |
| :----------------------------------------------------------: | :-------------: | :-------------: | :-------------: | :---------: |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |       20        |     2698.6      |     20756.0     |    1800     |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |       18        |     2911.6      |     20988.0     |    1800     |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |       18        |     2701.0      |     20904.1     |    1800     |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |       18        |     2637.2      |     20751.5     |    1800     |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |       20        |     2694.9      |     20694.9     |    1800     |
|                            c1_8_1                            |       80        |     25156.9     |     97592.4     |    1800     |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |       74        |     25501.7     |     98696.5     |    1800     |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |       72        |     24431.7     |     97769.7     |    1800     |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |       72        |     23944.2     |     97420.8     |    1800     |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |       80        |     25138.6     |     97516.5     |    1800     |

## Environments

- Operating System: Windows 11（本地测试），Linux（主办方服务器）
- 8核 + 256G 的配置



## How to Start?

`python ./src/main.py <Instance file path> --runtime <Maxtime>`

Example:

`python ./src/main.py data/homberger_200_customer_instances/C1_2_1.TXT --runtime 30`

`python ./src/main.py data/homberger_200_customer_instances/C1_2_1.TXT --runtime 1800 --runs 10 --num_subproblems 2`

`python ./src/main.py data/homberger_800_customer_instances/C1_8_1.TXT --runtime 1800 --runs 10 --num_subproblems 8`

## References

BKS:

- `[DOCS 2025 VRPTW track]` fleet minimization in priority, and then route length minimization: [https://www.sintef.no/projectweb/top/vrptw/200-customers/](https://www.sintef.no/projectweb/top/vrptw/200-customers/)

- `[DIMACS 2022 VRPTW track]` minimizing distances, not the number of vehicles used: [http://vrp.galgos.inf.puc-rio.br/index.php/en/](http://vrp.galgos.inf.puc-rio.br/index.php/en/) (with optimality provided)

PyVRP:



## Questions

- [x] 计算精度：坐标间的欧几里得距离应该保留几位小数（Ours默认一位小数）？目前表格Distance比较没有意义。根据VRPTW.md文件示例输出，Distance**默认是一位小数**，主办方在微信群 [2025/07/08] 最新通知是**直接一位截断小数**
- [x] 读入文件处理：目前是粗糙地将Solomon格式转化为CVRPLIB格式读入，但是这种转换默认每个客户的SERVICE_TIME相同。后面**要做好更加规范严格的数据读入**。解决方案：重构PyVRP库里的read.py，使之支持读入Solomon格式Instance。
- [x] 优化目标修改：目前是直接最小化Distance，因此：我要修改使得优先最小化车辆数，再最小化Distance。解决方案：将fixed_costs设置为10000
- [x] 创新改进：分解策略 barycenter clustering decomposition + subproblem parallel computing with 8 CPU cores

- [ ] 主办方服务器上测试算法
