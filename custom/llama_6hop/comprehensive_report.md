# Comprehensive Causal Tracking Experiment Report

This report details the accuracy of models on tracking objects across 1 to 5 hops, separated by category (Living vs. Non-Living), prompt structure (Normal vs. Shifted distractor), and varying neutral/semantic verbs.

## Model: meta-llama/Llama-3.2-1B
---

### Hop Count: 1

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 5/10 (50%) | 4/10 (40%) | 3/10 (30%) | 4/10 (40%) |
| `is with the` | 7/10 (70%) | 4/10 (40%) | 3/10 (30%) | 4/10 (40%) |
| `is carried by the` | 6/10 (60%) | 3/10 (30%) | 2/10 (20%) | 6/10 (60%) |
| `is in the possession of the` | 7/10 (70%) | 4/10 (40%) | 3/10 (30%) | 4/10 (40%) |
| `is currently with the` | 8/10 (80%) | 4/10 (40%) | 2/10 (20%) | 7/10 (70%) |
| `belongs to the` | 4/10 (40%) | 6/10 (60%) | 3/10 (30%) | 2/10 (20%) |
| `is near the` | 6/10 (60%) | 2/10 (20%) | 3/10 (30%) | 3/10 (30%) |
| `is beside the` | 5/10 (50%) | 1/10 (10%) | 2/10 (20%) | 2/10 (20%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 5/10 (50%) | 3/10 (30%) | 3/10 (30%) | 5/10 (50%) |
| `is inside the` | 5/10 (50%) | 0/10 (0%) | 0/10 (0%) | 5/10 (50%) |
| `is placed in the` | 6/10 (60%) | 2/10 (20%) | 4/10 (40%) | 3/10 (30%) |
| `is resting in the` | 5/10 (50%) | 1/10 (10%) | 4/10 (40%) | 6/10 (60%) |
| `is found in the` | 5/10 (50%) | 3/10 (30%) | 3/10 (30%) | 6/10 (60%) |
| `is in the` | 7/10 (70%) | 2/10 (20%) | 2/10 (20%) | 7/10 (70%) |
| `is at the` | 5/10 (50%) | 2/10 (20%) | 3/10 (30%) | 4/10 (40%) |
| `is near the` | 4/10 (40%) | 3/10 (30%) | 2/10 (20%) | 4/10 (40%) |

### Hop Count: 2

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 4/10 (40%) |
| `is with the` | 8/10 (80%) | 2/10 (20%) | 0/10 (0%) | 6/10 (60%) |
| `is carried by the` | 9/10 (90%) | 0/10 (0%) | 0/10 (0%) | 7/10 (70%) |
| `is in the possession of the` | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 4/10 (40%) |
| `is currently with the` | 9/10 (90%) | 1/10 (10%) | 0/10 (0%) | 4/10 (40%) |
| `belongs to the` | 6/10 (60%) | 2/10 (20%) | 1/10 (10%) | 1/10 (10%) |
| `is near the` | 6/10 (60%) | 2/10 (20%) | 0/10 (0%) | 3/10 (30%) |
| `is beside the` | 6/10 (60%) | 1/10 (10%) | 0/10 (0%) | 2/10 (20%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 7/10 (70%) | 2/10 (20%) | 2/10 (20%) | 8/10 (80%) |
| `is inside the` | 6/10 (60%) | 3/10 (30%) | 1/10 (10%) | 8/10 (80%) |
| `is placed in the` | 8/10 (80%) | 3/10 (30%) | 2/10 (20%) | 3/10 (30%) |
| `is resting in the` | 6/10 (60%) | 4/10 (40%) | 2/10 (20%) | 10/10 (100%) |
| `is found in the` | 7/10 (70%) | 3/10 (30%) | 2/10 (20%) | 8/10 (80%) |
| `is in the` | 7/10 (70%) | 4/10 (40%) | 1/10 (10%) | 10/10 (100%) |
| `is at the` | 5/10 (50%) | 1/10 (10%) | 3/10 (30%) | 3/10 (30%) |
| `is near the` | 6/10 (60%) | 1/10 (10%) | 1/10 (10%) | 5/10 (50%) |

### Hop Count: 3

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 2/10 (20%) | 1/10 (10%) | 3/10 (30%) |
| `is with the` | 8/10 (80%) | 2/10 (20%) | 1/10 (10%) | 6/10 (60%) |
| `is carried by the` | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 6/10 (60%) |
| `is in the possession of the` | 10/10 (100%) | 1/10 (10%) | 2/10 (20%) | 2/10 (20%) |
| `is currently with the` | 10/10 (100%) | 1/10 (10%) | 3/10 (30%) | 4/10 (40%) |
| `belongs to the` | 8/10 (80%) | 5/10 (50%) | 0/10 (0%) | 0/10 (0%) |
| `is near the` | 8/10 (80%) | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) |
| `is beside the` | 6/10 (60%) | 0/10 (0%) | 0/10 (0%) | 2/10 (20%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 8/10 (80%) | 0/10 (0%) | 2/10 (20%) | 9/10 (90%) |
| `is inside the` | 9/10 (90%) | 0/10 (0%) | 1/10 (10%) | 8/10 (80%) |
| `is placed in the` | 7/10 (70%) | 0/10 (0%) | 1/10 (10%) | 5/10 (50%) |
| `is resting in the` | 7/10 (70%) | 1/10 (10%) | 0/10 (0%) | 10/10 (100%) |
| `is found in the` | 9/10 (90%) | 0/10 (0%) | 1/10 (10%) | 10/10 (100%) |
| `is in the` | 9/10 (90%) | 1/10 (10%) | 1/10 (10%) | 9/10 (90%) |
| `is at the` | 9/10 (90%) | 0/10 (0%) | 2/10 (20%) | 2/10 (20%) |
| `is near the` | 7/10 (70%) | 0/10 (0%) | 0/10 (0%) | 6/10 (60%) |

### Hop Count: 4

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 9/10 (90%) | 0/10 (0%) | 0/10 (0%) | 5/10 (50%) |
| `is with the` | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 6/10 (60%) |
| `is carried by the` | 9/10 (90%) | 1/10 (10%) | 0/10 (0%) | 9/10 (90%) |
| `is in the possession of the` | 9/10 (90%) | 0/10 (0%) | 0/10 (0%) | 5/10 (50%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 7/10 (70%) |
| `belongs to the` | 6/10 (60%) | 2/10 (20%) | 0/10 (0%) | 1/10 (10%) |
| `is near the` | 5/10 (50%) | 0/10 (0%) | 0/10 (0%) | 2/10 (20%) |
| `is beside the` | 7/10 (70%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 9/10 (90%) | 1/10 (10%) | 0/10 (0%) | 9/10 (90%) |
| `is inside the` | 8/10 (80%) | 2/10 (20%) | 0/10 (0%) | 8/10 (80%) |
| `is placed in the` | 8/10 (80%) | 3/10 (30%) | 1/10 (10%) | 7/10 (70%) |
| `is resting in the` | 8/10 (80%) | 3/10 (30%) | 0/10 (0%) | 10/10 (100%) |
| `is found in the` | 8/10 (80%) | 3/10 (30%) | 0/10 (0%) | 10/10 (100%) |
| `is in the` | 9/10 (90%) | 2/10 (20%) | 0/10 (0%) | 9/10 (90%) |
| `is at the` | 8/10 (80%) | 1/10 (10%) | 1/10 (10%) | 6/10 (60%) |
| `is near the` | 7/10 (70%) | 0/10 (0%) | 0/10 (0%) | 6/10 (60%) |

### Hop Count: 5

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 1/10 (10%) | 0/10 (0%) | 5/10 (50%) |
| `is with the` | 9/10 (90%) | 1/10 (10%) | 0/10 (0%) | 8/10 (80%) |
| `is carried by the` | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 7/10 (70%) |
| `is in the possession of the` | 9/10 (90%) | 0/10 (0%) | 0/10 (0%) | 6/10 (60%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 6/10 (60%) |
| `belongs to the` | 7/10 (70%) | 2/10 (20%) | 1/10 (10%) | 2/10 (20%) |
| `is near the` | 7/10 (70%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |
| `is beside the` | 8/10 (80%) | 0/10 (0%) | 0/10 (0%) | 4/10 (40%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 7/10 (70%) | 1/10 (10%) | 1/10 (10%) | 9/10 (90%) |
| `is inside the` | 7/10 (70%) | 0/10 (0%) | 0/10 (0%) | 9/10 (90%) |
| `is placed in the` | 7/10 (70%) | 2/10 (20%) | 0/10 (0%) | 2/10 (20%) |
| `is resting in the` | 7/10 (70%) | 1/10 (10%) | 0/10 (0%) | 8/10 (80%) |
| `is found in the` | 7/10 (70%) | 2/10 (20%) | 1/10 (10%) | 9/10 (90%) |
| `is in the` | 7/10 (70%) | 2/10 (20%) | 0/10 (0%) | 9/10 (90%) |
| `is at the` | 8/10 (80%) | 0/10 (0%) | 1/10 (10%) | 1/10 (10%) |
| `is near the` | 6/10 (60%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |

## Model: meta-llama/Meta-Llama-3.1-8B
---

### Hop Count: 1

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 5/10 (50%) | 2/10 (20%) | 1/10 (10%) | 4/10 (40%) |
| `is with the` | 6/10 (60%) | 4/10 (40%) | 1/10 (10%) | 5/10 (50%) |
| `is carried by the` | 4/10 (40%) | 2/10 (20%) | 1/10 (10%) | 5/10 (50%) |
| `is in the possession of the` | 5/10 (50%) | 3/10 (30%) | 1/10 (10%) | 5/10 (50%) |
| `is currently with the` | 7/10 (70%) | 2/10 (20%) | 0/10 (0%) | 6/10 (60%) |
| `belongs to the` | 5/10 (50%) | 4/10 (40%) | 0/10 (0%) | 2/10 (20%) |
| `is near the` | 2/10 (20%) | 3/10 (30%) | 1/10 (10%) | 1/10 (10%) |
| `is beside the` | 0/10 (0%) | 2/10 (20%) | 0/10 (0%) | 2/10 (20%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 2/10 (20%) | 7/10 (70%) | 0/10 (0%) | 5/10 (50%) |
| `is inside the` | 2/10 (20%) | 5/10 (50%) | 0/10 (0%) | 7/10 (70%) |
| `is placed in the` | 3/10 (30%) | 6/10 (60%) | 0/10 (0%) | 1/10 (10%) |
| `is resting in the` | 2/10 (20%) | 8/10 (80%) | 0/10 (0%) | 10/10 (100%) |
| `is found in the` | 2/10 (20%) | 9/10 (90%) | 0/10 (0%) | 9/10 (90%) |
| `is in the` | 2/10 (20%) | 5/10 (50%) | 0/10 (0%) | 5/10 (50%) |
| `is at the` | 2/10 (20%) | 4/10 (40%) | 1/10 (10%) | 1/10 (10%) |
| `is near the` | 3/10 (30%) | 2/10 (20%) | 0/10 (0%) | 2/10 (20%) |

### Hop Count: 2

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 9/10 (90%) | 0/10 (0%) | 0/10 (0%) | 6/10 (60%) |
| `is with the` | 10/10 (100%) | 1/10 (10%) | 1/10 (10%) | 4/10 (40%) |
| `is carried by the` | 8/10 (80%) | 0/10 (0%) | 1/10 (10%) | 6/10 (60%) |
| `is in the possession of the` | 9/10 (90%) | 1/10 (10%) | 1/10 (10%) | 4/10 (40%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 5/10 (50%) |
| `belongs to the` | 7/10 (70%) | 2/10 (20%) | 1/10 (10%) | 2/10 (20%) |
| `is near the` | 3/10 (30%) | 4/10 (40%) | 2/10 (20%) | 2/10 (20%) |
| `is beside the` | 1/10 (10%) | 3/10 (30%) | 2/10 (20%) | 2/10 (20%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 6/10 (60%) | 3/10 (30%) | 3/10 (30%) | 2/10 (20%) |
| `is inside the` | 5/10 (50%) | 3/10 (30%) | 0/10 (0%) | 10/10 (100%) |
| `is placed in the` | 8/10 (80%) | 2/10 (20%) | 1/10 (10%) | 0/10 (0%) |
| `is resting in the` | 7/10 (70%) | 2/10 (20%) | 4/10 (40%) | 10/10 (100%) |
| `is found in the` | 7/10 (70%) | 3/10 (30%) | 2/10 (20%) | 7/10 (70%) |
| `is in the` | 6/10 (60%) | 3/10 (30%) | 0/10 (0%) | 8/10 (80%) |
| `is at the` | 5/10 (50%) | 1/10 (10%) | 0/10 (0%) | 0/10 (0%) |
| `is near the` | 1/10 (10%) | 0/10 (0%) | 2/10 (20%) | 0/10 (0%) |

### Hop Count: 3

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 3/10 (30%) |
| `is with the` | 10/10 (100%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |
| `is carried by the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 5/10 (50%) |
| `is in the possession of the` | 10/10 (100%) | 0/10 (0%) | 2/10 (20%) | 3/10 (30%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 2/10 (20%) | 4/10 (40%) |
| `belongs to the` | 6/10 (60%) | 2/10 (20%) | 1/10 (10%) | 1/10 (10%) |
| `is near the` | 4/10 (40%) | 0/10 (0%) | 2/10 (20%) | 0/10 (0%) |
| `is beside the` | 4/10 (40%) | 0/10 (0%) | 2/10 (20%) | 0/10 (0%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 7/10 (70%) | 2/10 (20%) | 2/10 (20%) | 4/10 (40%) |
| `is inside the` | 7/10 (70%) | 0/10 (0%) | 0/10 (0%) | 8/10 (80%) |
| `is placed in the` | 5/10 (50%) | 1/10 (10%) | 1/10 (10%) | 0/10 (0%) |
| `is resting in the` | 9/10 (90%) | 2/10 (20%) | 1/10 (10%) | 7/10 (70%) |
| `is found in the` | 8/10 (80%) | 2/10 (20%) | 2/10 (20%) | 6/10 (60%) |
| `is in the` | 9/10 (90%) | 2/10 (20%) | 1/10 (10%) | 6/10 (60%) |
| `is at the` | 6/10 (60%) | 0/10 (0%) | 1/10 (10%) | 0/10 (0%) |
| `is near the` | 3/10 (30%) | 0/10 (0%) | 1/10 (10%) | 0/10 (0%) |

### Hop Count: 4

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 9/10 (90%) |
| `is with the` | 8/10 (80%) | 0/10 (0%) | 0/10 (0%) | 5/10 (50%) |
| `is carried by the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 9/10 (90%) |
| `is in the possession of the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 7/10 (70%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 1/10 (10%) | 8/10 (80%) |
| `belongs to the` | 8/10 (80%) | 2/10 (20%) | 1/10 (10%) | 4/10 (40%) |
| `is near the` | 6/10 (60%) | 1/10 (10%) | 2/10 (20%) | 0/10 (0%) |
| `is beside the` | 5/10 (50%) | 0/10 (0%) | 1/10 (10%) | 0/10 (0%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 7/10 (70%) | 2/10 (20%) | 1/10 (10%) | 4/10 (40%) |
| `is inside the` | 8/10 (80%) | 0/10 (0%) | 0/10 (0%) | 9/10 (90%) |
| `is placed in the` | 5/10 (50%) | 1/10 (10%) | 0/10 (0%) | 1/10 (10%) |
| `is resting in the` | 6/10 (60%) | 2/10 (20%) | 0/10 (0%) | 9/10 (90%) |
| `is found in the` | 9/10 (90%) | 2/10 (20%) | 1/10 (10%) | 7/10 (70%) |
| `is in the` | 8/10 (80%) | 2/10 (20%) | 0/10 (0%) | 7/10 (70%) |
| `is at the` | 6/10 (60%) | 1/10 (10%) | 0/10 (0%) | 2/10 (20%) |
| `is near the` | 3/10 (30%) | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) |

### Hop Count: 5

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 1/10 (10%) | 0/10 (0%) | 6/10 (60%) |
| `is with the` | 9/10 (90%) | 0/10 (0%) | 1/10 (10%) | 5/10 (50%) |
| `is carried by the` | 9/10 (90%) | 0/10 (0%) | 1/10 (10%) | 6/10 (60%) |
| `is in the possession of the` | 10/10 (100%) | 0/10 (0%) | 2/10 (20%) | 5/10 (50%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 2/10 (20%) | 6/10 (60%) |
| `belongs to the` | 7/10 (70%) | 1/10 (10%) | 2/10 (20%) | 2/10 (20%) |
| `is near the` | 5/10 (50%) | 2/10 (20%) | 1/10 (10%) | 2/10 (20%) |
| `is beside the` | 4/10 (40%) | 0/10 (0%) | 0/10 (0%) | 2/10 (20%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 7/10 (70%) | 0/10 (0%) | 1/10 (10%) | 6/10 (60%) |
| `is inside the` | 6/10 (60%) | 1/10 (10%) | 1/10 (10%) | 10/10 (100%) |
| `is placed in the` | 6/10 (60%) | 0/10 (0%) | 1/10 (10%) | 2/10 (20%) |
| `is resting in the` | 7/10 (70%) | 2/10 (20%) | 1/10 (10%) | 10/10 (100%) |
| `is found in the` | 9/10 (90%) | 1/10 (10%) | 1/10 (10%) | 7/10 (70%) |
| `is in the` | 7/10 (70%) | 1/10 (10%) | 1/10 (10%) | 10/10 (100%) |
| `is at the` | 8/10 (80%) | 1/10 (10%) | 1/10 (10%) | 0/10 (0%) |
| `is near the` | 5/10 (50%) | 0/10 (0%) | 0/10 (0%) | 2/10 (20%) |

## Model: Qwen/Qwen2.5-32B
---

### Hop Count: 1

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 6/10 (60%) | 3/10 (30%) | 6/10 (60%) | 8/10 (80%) |
| `is with the` | 8/10 (80%) | 3/10 (30%) | 4/10 (40%) | 7/10 (70%) |
| `is carried by the` | 5/10 (50%) | 2/10 (20%) | 3/10 (30%) | 7/10 (70%) |
| `is in the possession of the` | 7/10 (70%) | 3/10 (30%) | 4/10 (40%) | 10/10 (100%) |
| `is currently with the` | 8/10 (80%) | 4/10 (40%) | 9/10 (90%) | 10/10 (100%) |
| `belongs to the` | 4/10 (40%) | 6/10 (60%) | 3/10 (30%) | 7/10 (70%) |
| `is near the` | 4/10 (40%) | 2/10 (20%) | 2/10 (20%) | 7/10 (70%) |
| `is beside the` | 3/10 (30%) | 0/10 (0%) | 0/10 (0%) | 4/10 (40%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 6/10 (60%) | 6/10 (60%) | 3/10 (30%) | 5/10 (50%) |
| `is inside the` | 4/10 (40%) | 4/10 (40%) | 1/10 (10%) | 5/10 (50%) |
| `is placed in the` | 2/10 (20%) | 1/10 (10%) | 4/10 (40%) | 0/10 (0%) |
| `is resting in the` | 5/10 (50%) | 8/10 (80%) | 2/10 (20%) | 10/10 (100%) |
| `is found in the` | 6/10 (60%) | 10/10 (100%) | 4/10 (40%) | 10/10 (100%) |
| `is in the` | 9/10 (90%) | 10/10 (100%) | 2/10 (20%) | 9/10 (90%) |
| `is at the` | 6/10 (60%) | 4/10 (40%) | 3/10 (30%) | 3/10 (30%) |
| `is near the` | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) |

### Hop Count: 2

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 8/10 (80%) | 1/10 (10%) | 4/10 (40%) | 9/10 (90%) |
| `is with the` | 8/10 (80%) | 0/10 (0%) | 2/10 (20%) | 7/10 (70%) |
| `is carried by the` | 6/10 (60%) | 1/10 (10%) | 2/10 (20%) | 9/10 (90%) |
| `is in the possession of the` | 9/10 (90%) | 0/10 (0%) | 8/10 (80%) | 10/10 (100%) |
| `is currently with the` | 9/10 (90%) | 0/10 (0%) | 9/10 (90%) | 10/10 (100%) |
| `belongs to the` | 4/10 (40%) | 3/10 (30%) | 4/10 (40%) | 9/10 (90%) |
| `is near the` | 5/10 (50%) | 4/10 (40%) | 0/10 (0%) | 7/10 (70%) |
| `is beside the` | 2/10 (20%) | 0/10 (0%) | 0/10 (0%) | 4/10 (40%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 8/10 (80%) | 8/10 (80%) | 7/10 (70%) | 6/10 (60%) |
| `is inside the` | 5/10 (50%) | 8/10 (80%) | 2/10 (20%) | 9/10 (90%) |
| `is placed in the` | 4/10 (40%) | 1/10 (10%) | 2/10 (20%) | 1/10 (10%) |
| `is resting in the` | 8/10 (80%) | 10/10 (100%) | 5/10 (50%) | 10/10 (100%) |
| `is found in the` | 9/10 (90%) | 8/10 (80%) | 7/10 (70%) | 10/10 (100%) |
| `is in the` | 9/10 (90%) | 9/10 (90%) | 5/10 (50%) | 10/10 (100%) |
| `is at the` | 6/10 (60%) | 5/10 (50%) | 4/10 (40%) | 1/10 (10%) |
| `is near the` | 0/10 (0%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |

### Hop Count: 3

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 0/10 (0%) | 5/10 (50%) | 9/10 (90%) |
| `is with the` | 10/10 (100%) | 0/10 (0%) | 7/10 (70%) | 7/10 (70%) |
| `is carried by the` | 8/10 (80%) | 0/10 (0%) | 2/10 (20%) | 9/10 (90%) |
| `is in the possession of the` | 10/10 (100%) | 0/10 (0%) | 7/10 (70%) | 10/10 (100%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 10/10 (100%) | 9/10 (90%) |
| `belongs to the` | 8/10 (80%) | 1/10 (10%) | 6/10 (60%) | 4/10 (40%) |
| `is near the` | 4/10 (40%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |
| `is beside the` | 2/10 (20%) | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 9/10 (90%) | 8/10 (80%) | 4/10 (40%) | 6/10 (60%) |
| `is inside the` | 5/10 (50%) | 8/10 (80%) | 2/10 (20%) | 8/10 (80%) |
| `is placed in the` | 5/10 (50%) | 3/10 (30%) | 2/10 (20%) | 1/10 (10%) |
| `is resting in the` | 7/10 (70%) | 9/10 (90%) | 4/10 (40%) | 10/10 (100%) |
| `is found in the` | 9/10 (90%) | 8/10 (80%) | 8/10 (80%) | 10/10 (100%) |
| `is in the` | 10/10 (100%) | 8/10 (80%) | 9/10 (90%) | 8/10 (80%) |
| `is at the` | 9/10 (90%) | 3/10 (30%) | 3/10 (30%) | 1/10 (10%) |
| `is near the` | 0/10 (0%) | 2/10 (20%) | 0/10 (0%) | 2/10 (20%) |

### Hop Count: 4

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 1/10 (10%) | 6/10 (60%) | 8/10 (80%) |
| `is with the` | 10/10 (100%) | 1/10 (10%) | 5/10 (50%) | 4/10 (40%) |
| `is carried by the` | 10/10 (100%) | 3/10 (30%) | 2/10 (20%) | 9/10 (90%) |
| `is in the possession of the` | 10/10 (100%) | 1/10 (10%) | 8/10 (80%) | 9/10 (90%) |
| `is currently with the` | 10/10 (100%) | 0/10 (0%) | 9/10 (90%) | 10/10 (100%) |
| `belongs to the` | 9/10 (90%) | 3/10 (30%) | 2/10 (20%) | 7/10 (70%) |
| `is near the` | 6/10 (60%) | 1/10 (10%) | 0/10 (0%) | 4/10 (40%) |
| `is beside the` | 5/10 (50%) | 0/10 (0%) | 0/10 (0%) | 2/10 (20%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 8/10 (80%) | 9/10 (90%) | 3/10 (30%) | 8/10 (80%) |
| `is inside the` | 6/10 (60%) | 5/10 (50%) | 2/10 (20%) | 7/10 (70%) |
| `is placed in the` | 5/10 (50%) | 1/10 (10%) | 0/10 (0%) | 0/10 (0%) |
| `is resting in the` | 9/10 (90%) | 6/10 (60%) | 5/10 (50%) | 10/10 (100%) |
| `is found in the` | 9/10 (90%) | 7/10 (70%) | 6/10 (60%) | 10/10 (100%) |
| `is in the` | 10/10 (100%) | 5/10 (50%) | 4/10 (40%) | 10/10 (100%) |
| `is at the` | 8/10 (80%) | 4/10 (40%) | 2/10 (20%) | 2/10 (20%) |
| `is near the` | 2/10 (20%) | 1/10 (10%) | 0/10 (0%) | 3/10 (30%) |

### Hop Count: 5

#### Category: Living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is held by the` | 10/10 (100%) | 0/10 (0%) | 4/10 (40%) | 7/10 (70%) |
| `is with the` | 10/10 (100%) | 1/10 (10%) | 6/10 (60%) | 7/10 (70%) |
| `is carried by the` | 9/10 (90%) | 1/10 (10%) | 4/10 (40%) | 7/10 (70%) |
| `is in the possession of the` | 10/10 (100%) | 0/10 (0%) | 7/10 (70%) | 8/10 (80%) |
| `is currently with the` | 9/10 (90%) | 0/10 (0%) | 6/10 (60%) | 10/10 (100%) |
| `belongs to the` | 10/10 (100%) | 2/10 (20%) | 3/10 (30%) | 6/10 (60%) |
| `is near the` | 8/10 (80%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |
| `is beside the` | 5/10 (50%) | 0/10 (0%) | 0/10 (0%) | 3/10 (30%) |

#### Category: Non_living
| Verb | Normal Clean | Normal Corrupted | Shifted Clean | Shifted Corrupted |
| :--- | :--- | :--- | :--- | :--- |
| `is located in the` | 8/10 (80%) | 7/10 (70%) | 3/10 (30%) | 4/10 (40%) |
| `is inside the` | 8/10 (80%) | 7/10 (70%) | 2/10 (20%) | 7/10 (70%) |
| `is placed in the` | 3/10 (30%) | 1/10 (10%) | 0/10 (0%) | 1/10 (10%) |
| `is resting in the` | 10/10 (100%) | 8/10 (80%) | 4/10 (40%) | 10/10 (100%) |
| `is found in the` | 10/10 (100%) | 8/10 (80%) | 4/10 (40%) | 10/10 (100%) |
| `is in the` | 10/10 (100%) | 8/10 (80%) | 3/10 (30%) | 9/10 (90%) |
| `is at the` | 7/10 (70%) | 2/10 (20%) | 3/10 (30%) | 1/10 (10%) |
| `is near the` | 1/10 (10%) | 0/10 (0%) | 0/10 (0%) | 1/10 (10%) |

