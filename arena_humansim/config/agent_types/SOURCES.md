# Sources of the mobility-aid agent types

One row per type and parameter. Types that share a value share a row. **Status:**
- *measured*: the value is taken from the study;
- *derived*: computed from a measured value, the calculation is given;
- *estimate*: no study found, so the value is our own. "Inherited" means the type keeps the value of the type it extends.

The `yaml` field is where the value lives in the agent-type file.

| Type | Parameter (`yaml` field) | Value | Unit | Source | Who was measured | Status |
|---|---|---|---|---|---|---|
| white cane, rolling cane | comfortable speed (`desired_velocity`) | 0.82 ± 0.24 | m/s | [Emerson 2019] Table 5, blind and low-vision participants: 0.820 ± 0.241 m/s (the overall 0.894 ± 0.197 of Table 4 includes blindfolded sighted participants, 0.950 ± 0.128). The pooled studies varied the technique (two-point touch, constant contact) and the tip (including a roller tip) but report no speed per technique, so the rolling cane shares the value | blind and low-vision long cane users, 4,324 lab trials of drop-off and obstacle detection | measured |
| white cane, rolling cane | maximum speed (`max_velocity`) | 1.1 ± 0.2 | m/s | none found; the adult's max/comfortable ratio 1.5/1.1 applied to the measured mean | — | estimate |
| white cane, rolling cane | stride | 1.14 ± 0.21 (sighted 1.39) | m | [Hallemans 2010] | adults with low vision | measured, not used by the config |
| white cane, rolling cane | acceleration (`max_acceleration`) | 0.8 ± 0.1 | m/s² | none | — | estimate |
| white cane, rolling cane | stopping (`max_deceleration`) | 2.5 ± 0.2 | m/s² | none | — | estimate, inherited from `adult` |
| white cane, rolling cane | sight (`vision_range`, `proximity_sense`) | 0 | m | type definition: no usable sight; hearing and the cane replace the proximity sense | — | estimate |
| white cane, rolling cane | cane reach ahead (`cane_reach`) | 1.05 | m | Arena cane model, `arena_animation_retarget/hold.py`: 1.27 m cane held 0.1 m below its top, fist 0.9 m high, 0.3 m ahead and 0.1 m right, so the tip lands 0.93–1.05 m ahead. A long cane is cut per user to reach the sternum; 5% shorter or 10% longer detects drop-offs worse [Emerson 2019, citing Rodgers & Emerson 2005], [Kim 2017] | — | derived |
| white cane, rolling cane | cane sweep width (`cane_width`) | 0.6 | m | Arena sweep (`hold.py`, ±0.30 m), so that sensing and collisions use one cane. Measured arcs: 0.83 ± 0.23 m, "slightly wider than a cane user's shoulder width" [Emerson 2019, Table 4]; 0.6 m is within one SD. The rolling cane keeps constant contact with the same reach and width | 101 cane users, 10,069 trials | derived |
| white cane, rolling cane | hearing a robot or a walker (`hearing_range`) | 2.0 | m | Robots are heard out to `hearing_range` × `robot_loudness` × speed / (1 m/s) and not below 0.02 m/s: the law of arena_auditory's drivetrain (45 dB at 1 m at 1 m/s, level in proportion to speed) with a level falling 6 dB per doubling of distance [arena-auditory]. 2.0 m at 1 m/s is an estimate: the same law gives 1.8 m with the 40 dBA ambient bed arena_auditory assumes and 17.8 m with its 20 dB default threshold. No study measures hearing robots; cars give an upper bound: [Emerson 2013] a quiet electric car at 15 km/h is heard at 49 ± 23 m, a combustion car at 78 m, in a quiet street; background noise halves the distance [Kim 2012] | blind adults, N = 11 (cars) | estimate |
| white cane, rolling cane | detection delay (`detection_delay`) | 0.5 | s | none found. [Kayukawa 2019] warns blind travellers 5 s before a predicted collision (a design value) | — | estimate |
| white cane, rolling cane | footprint (`agent_radius`) | 0.25 ± 0.02 | m | body only. The cane is not part of the disc: Arena projects the posed cane to 2D for collisions | — | estimate, inherited from `adult` |
| rollator | comfortable speed (`desired_velocity`) | 0.56 ± 0.18 | m/s | [Werner 2020] Table 2: rollator-assisted walking at self-selected speed over a 10 m walkway. Fit older adults are faster: 1.07 m/s on the level, 0.79 m/s uphill [Mundt 2019] | 58 acutely hospitalized geriatric patients with gait and balance disorders, 83.7 ± 5.9 years, all using a rollator | measured |
| rollator | maximum speed (`max_velocity`) | 0.76 ± 0.1 | m/s | none found; the adult's max/comfortable ratio 1.5/1.1 applied to the measured mean | — | estimate |
| rollator | acceleration (`max_acceleration`) | 0.5 ± 0.1 | m/s² | none | — | estimate |
| rollator | stopping (`max_deceleration`) | 0.8 ± 0.1 | m/s² | none | — | estimate |
| rollator | footprint (`footprint_length`, `agent_radius`) | 1.0 / 0.32 | m | RollatorFitted object 0.571 × 0.689 m | — | derived |
| walking frame | comfortable speed (`desired_velocity`) | 0.17 ± 0.02 | m/s | none found | — | estimate |
| walking frame | maximum speed (`max_velocity`) | 0.25 ± 0.03 | m/s | none | — | estimate |
| walking frame | acceleration (`max_acceleration`) | 0.3 ± 0.05 | m/s² | none | — | estimate |
| walking frame | stopping (`max_deceleration`) | 0.6 ± 0.1 | m/s² | none | — | estimate |
| walking frame | pauses | not modelled | | no measured pause data found | — | — |
| walking frame | footprint (`footprint_length`, `agent_radius`) | 0.95 / 0.32 | m | none | — | estimate |
| IV stand | comfortable speed (`desired_velocity`) | 0.6 ± 0.05 | m/s | none found | — | estimate |
| IV stand | maximum speed (`max_velocity`) | 0.75 ± 0.05 | m/s | none | — | estimate |
| IV stand | acceleration (`max_acceleration`) | 0.6 ± 0.1 | m/s² | none | — | estimate |
| IV stand | stopping (`max_deceleration`) | 1.5 ± 0.2 | m/s² | none | — | estimate, inherited from `elder` |
| IV stand | footprint (`agent_radius`) | 0.38 ± 0.02 | m | none | — | estimate |
| rollator, walking frame, IV stand | sight (`vision_range`, `vision_fov`) | 3.5 ± 0.8 m, 140 ± 15° | | none | — | estimate, inherited from `elder` |
| rollator, walking frame, IV stand | detection delay (`detection_delay`) | 0.5 | s | older adults start avoiding later and pass closer: 0.79 ± 0.18 m versus 0.93 ± 0.17 m between young adults [Rapos 2021] | older adults about 68 years, young adults about 23 | estimate |
| limp | comfortable speed (`desired_velocity`) | 0.86 ± 0.12 | m/s | adult 1.1 m/s × 0.78: hip OA walks 22% slower than controls (−0.26 m/s, 95% CI −0.32 to −0.20; controls 1.24 m/s) [Baily 2023]; an earlier review found 26% [Constantinou 2014]. The SD is the adult's | meta-analysis of 33 studies | derived |
| limp | maximum speed (`max_velocity`) | 1.17 ± 0.1 | m/s | adult 1.5 m/s × 0.78, same source | | derived |
| limp | acceleration, stopping (`max_acceleration`, `max_deceleration`) | 1.5 ± 0.1, 2.5 ± 0.2 | m/s² | none | — | estimate, inherited from `adult` |
| limp | footprint (`agent_radius`) | 0.25 ± 0.02 | m | none | — | estimate, inherited from `adult` |
| self-propelled wheelchair | comfortable speed (`desired_velocity`) | 1.04 ± 0.30 | m/s | [Slowik 2015] self-selected speed on the level (§2.2). Overground in a rehabilitation hospital: 1.17 ± 0.35 m/s [Coutinho 2013, abstract]; everyday bouts are slower, 0.48 ± 0.21 m/s [Sonenblum 2012] | 170 manual wheelchair users with paraplegia, on an ergometer | measured |
| self-propelled wheelchair | maximum speed (`max_velocity`) | 1.90 ± 0.46 | m/s | [Slowik 2015] fastest comfortable speed | as above | measured |
| self-propelled wheelchair | acceleration (`max_acceleration`) | 1.2 ± 0.1 | m/s² | none | — | estimate |
| self-propelled wheelchair | stopping (`max_deceleration`) | 2.5 ± 0.2 | m/s² | none | — | estimate, inherited from `adult` |
| self-propelled wheelchair | turning (`min_turning_radius`) | 0.05 | m | pivots in place; turning needs a 1.525 m circle [ADA 2010 §304.3], which the 1.1 m footprint sweeps | — | derived |
| self-propelled wheelchair | footprint (`footprint_length`, `agent_radius`) | 1.1 / 0.325 | m | none | — | estimate |
| pushed wheelchair | speed (`push_speed`, code default) | 0.9 | m/s | attendants pushing an NHS chair choose 0.8–1.12 m/s under a light load and 0.35–0.59 m/s under the heaviest [Suzuki 2015]; 0.9 is within the light-load range | a few healthy attendants | measured range |
| pushed wheelchair | motion limits and turning | as the self-propelled wheelchair | | the pushed chair is a `wheelchair_manual` agent; the caregiver follows at its handles (`formation/push.py`) | — | as above |
| pushed wheelchair | footprint | the chair, plus the caregiver 0.715 m behind its centre (`handle_offset`) | m | `formation/push.py` | — | estimate |
| limp, both wheelchairs | sight (`vision_range`, `vision_fov`) | 5.0 ± 0.5 m, 180 ± 10° | | none | — | estimate, inherited from `adult` |
| every type | reaction time (`reaction_time`) | 0.4, log-normal SD 0.3, clipped to 0.05–1.5 | s | code default, used only for scripted stimuli, not for robots. [Der & Deary 2006] not yet read | — | estimate |
| every type | clearance kept | not set; it comes out of the local planner | m | [Rapos 2021]: older adults pass young adults at 0.79 ± 0.18 m, two young adults at 0.93 ± 0.17 m | older adults about 68 years, young adults about 23 | measured, not used by the config |
| every type | yield or wait | yields (`robot_awareness`) | | people give way to a robot that does not react, even when they would pass first [Vassallo 2017] | adults | measured, behaviour only |

## References

- **[ADA 2010]** 2010 ADA Standards for Accessible Design, §304 Turning space. https://www.access-board.gov/ada/guides/chapter-3-clear-floor-or-ground-space-and-turning-space
- **[arena-auditory]** voshch/arena-auditory, jazzy: README and `arena_auditory/sources/drivetrain/spec.py` (motor level, `speed_exponent` 1, silent below 0.02 m/s), `params.py` (`propagation.threshold_db` 20, `render.bed.level_dba` 40). https://github.com/voshch/arena-auditory
- **[Baily 2023]** Baily A, Subasinghe Arachchige RSS, Constantinou M, Cheung RTH. Differences in temporal spatial gait characteristics between people with and without hip osteoarthritis: a systematic review with meta-analysis. JOSPT Open 1(1):16–33, 2023. doi:10.2519/josptopen.2023.0017
- **[Constantinou 2014]** Constantinou M, Barrett R, Brown M, Mills P. Spatial-temporal gait characteristics in individuals with hip osteoarthritis: a systematic literature review and meta-analysis. J Orthop Sports Phys Ther, 2014. doi:10.2519/jospt.2014.4634
- **[Coutinho 2013]** Coutinho AC, Neto FR, Perna CE. Determination of normative values for 20 min exercise of wheelchair propulsion by spinal cord injury patients. Spinal Cord 51(10):755–760, 2013. doi:10.1038/sc.2013.89
- **[Der & Deary 2006]** Der G, Deary IJ. Age and sex differences in reaction time in adulthood: results from the United Kingdom Health and Lifestyle Survey. Psychol Aging 21(1), 2006. https://pubmed.ncbi.nlm.nih.gov/16594792/
- **[Emerson 2013]** Emerson RW, Kim DS, Naghshineh K, Pliskow J, Myers K. Detection of quiet vehicles by blind pedestrians. J Transp Eng 139(1), 2013. doi:10.1061/(ASCE)TE.1943-5436.0000478
- **[Emerson 2019]** Emerson RW, et al. Biomechanics of long cane use. J Vis Impair Blind 113(3):235–247, 2019. doi:10.1177/0145482x19854928
- **[Hallemans 2010]** Hallemans A, Ortibus E, Meire F, Aerts P. Low vision affects dynamic stability of gait. Gait Posture, 2010. doi:10.1016/j.gaitpost.2010.07.018
- **[Kayukawa 2019]** Kayukawa S, Higuchi K, Guerreiro J, Morishima S, Sato Y, Kitani K, Asakawa C. BBeep: a sonic collision avoidance system for blind travellers and nearby pedestrians. CHI 2019. https://dblp.org/rec/conf/chi/KayukawaHGMSKA19
- **[Kim 2012]** Kim DS, Emerson RW, et al. Impact of adding artificially generated alert sound to hybrid electric vehicles on their detectability by pedestrians who are blind. J Rehabil Res Dev 49(3), 2012.
- **[Kim 2017]** Kim DS, Wall Emerson R, Naghshineh K. Effect of cane length and swing arc width on drop-off and obstacle detection with the long cane. Br J Vis Impair 35:217–231, 2017. doi:10.1177/0264619617700936
- **[Mundt 2019]** Mundt M, et al. Walking with rollator: a systematic review of gait parameters in older persons. Eur Rev Aging Phys Act 16:15, 2019. doi:10.1186/s11556-019-0222-5
- **[Rapos 2021]** Rapos V, Cinelli ME, Grunberg R, Bourgaize S, Crétual A, Olivier AH. Collision avoidance behaviours between older adult and young adult walkers. Gait Posture 88, 2021. doi:10.1016/j.gaitpost.2021.05.033
- **[Slowik 2015]** Slowik JS, Requejo PS, Mulroy SJ, Neptune RR. The influence of speed and grade on wheelchair propulsion hand pattern. Clin Biomech, 2015. doi:10.1016/j.clinbiomech.2015.07.007
- **[Sonenblum 2012]** Sonenblum SE, Sprigle S, Lopez RA. Manual wheelchair use: bouts of mobility in everyday life. Rehabil Res Pract 2012:753165. doi:10.1155/2012/753165
- **[Suzuki 2015]** Suzuki T, Utsuno H, Tyler N, Boampong D, Hayrapetyan K, Holloway C. Propelling load of an attendant propelled wheelchair in ascending and descending. J Biomech Sci Eng 10(2), 2015. https://www.jstage.jst.go.jp/article/jbse/10/2/10_14-00439/_article
- **[Vassallo 2017]** Vassallo C, Olivier AH, Souères P, Crétual A, Stasse O, Pettré J. How do walkers avoid a mobile robot crossing their way? Gait Posture 51, 2017. arXiv:1609.07955
- **[Werner 2020]** Werner C, Heldmann P, Hummel S, Bauknecht L, Bauer JM, Hauer K. Concurrent validity, test-retest reliability, and sensitivity to change of a single body-fixed sensor for gait analysis during rollator-assisted walking in acute geriatric patients. Sensors 20(17):4866, 2020. doi:10.3390/s20174866
