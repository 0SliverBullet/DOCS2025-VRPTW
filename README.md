# DOCS2025-VRPTW

Team: Route Seekers

## Experimental results

- 200-customer Instances

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Time (Ours) | Vehicles | Distance | Time | Reference | Date      | Comment                       |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------: | :------: | :------: | :--: | --------- | --------- | ----------------------------- |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |     **20**      |     2698.6      |     30      |  **20**  | 2704.57  |      | GH        | 2001      | Detailed solution by SAM::OPT |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |       20        |     2694.3      |     30      |  **18**  | 2917.89  |      | BVH       | 2001      | Detailed solution by SCR      |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |       20        |     2675.8      |     30      |  **18**  | 2707.35  |      | BSJ2      | 20-sep-07 | Detailed solution by SCR      |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |       19        |     2627.2      |     30      |  **18**  | 2643.31  |      | BSJ2      | 20-sep-07 | Detailed solution by SCR      |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |     **20**      |     2694.9      |     30      |  **20**  | 2702.05  |      | GH        | 2001      | Detailed solution by SAM::OPT |

- 800-customer Instances

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Time (Ours) | Vehicles | Distance | Time | Reference | Date                                                         |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------: | :------: | :------: | :--: | --------- | ------------------------------------------------------------ |
|                            c1_8_1                            |     **80**      |     25156.9     |     30      |  **80**  | 25030.36 |      | M         | 2002. There are questions whether the solution is valid, several authors report [80/25184.38](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_1.25184.38.sintef.txt) |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |       79        |     25069.3     |     30      |  **72**  | 26540.53 |      | CAINIAO   | Feb-19                                                       |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |       76        |     24890.5     |     30      |  **72**  | 24242.49 |      | SCR       | Oct-18                                                       |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |       73        |     24248.8     |     30      |  **72**  | 23824.17 |      | Q         | 28-oct-14                                                    |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |     **80**      |     25144.2     |     30      |  **80**  | 25166.28 |      | RP        | 25-feb-05                                                    |

Note: time in seconds.

- Improved Results

|                           Instance                           | Vehicles (Ours) | Distance (Ours) | Time (Ours) |
| :----------------------------------------------------------: | :-------------: | :-------------: | :---------: |
| [c1_2_1](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_1.txt) |                 |                 |             |
| [c1_2_2](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_2.18_2917.89.txt) |                 |                 |             |
| [c1_2_3](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |                 |                 |             |
| [c1_2_4](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_4.18_2643.31.txt) |                 |                 |             |
| [c1_2_5](https://www.sintef.no/contentassets/67388a7eea5c43cca4f52312c0688142/c1_2_5.txt) |                 |                 |             |
|                            c1_8_1                            |                 |                 |             |
| [c1_8_2](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_2-72-26540.53.txt) |                 |                 |             |
| [c1_8_3](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_3.72_24242.49.txt) |                 |                 |             |
| [c1_8_4](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_4-23824-17-sintef.txt) |                 |                 |             |
| [c1_8_5](https://www.sintef.no/contentassets/7951fb4f7ba04b7580ddcd23bd532cc1/c1_8_5.25166.28.sintef.txt) |                 |                 |             |



## References

BKS:

- `[DOCS 2025 VRPTW track]` fleet minimization in priority, and then route length minimization: [https://www.sintef.no/projectweb/top/vrptw/200-customers/](https://www.sintef.no/projectweb/top/vrptw/200-customers/)

- `[DIMACS 2022 VRPTW track]` minimizing distances, not the number of vehicles used: [http://vrp.galgos.inf.puc-rio.br/index.php/en/](http://vrp.galgos.inf.puc-rio.br/index.php/en/) (with optimality provided)

PyVRP:



## Questions

- [ ] 计算精度：坐标间的欧几里得距离应该保留几位小数（Ours默认一位小数）？目前表格Distance比较没有意义。根据VRPTW.md文件示例输出，Distance**默认是一位小数**，主办方说没有计算精度限制（逆天），那我摆了，**直接一位小数**
- [x] 读入文件处理：目前是粗糙地将Solomon格式转化为CVRPLIB格式读入，但是这种转换默认每个客户的SERVICE_TIME相同。后面**要做好更加规范严格的数据读入**。解决方案：重构PyVRP库里的read.py，使之支持读入Solomon格式Instance。

