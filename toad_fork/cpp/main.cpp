#include <cstring>
#include <algorithm>

extern "C" {

struct Unit {
  int id;
  int x;
  int y;
  int energy;
  int is_fake;
};

struct Relic {
  int player_id;
  int x;
  int y;
};

struct MyClass {
  Unit units_0[16];
  int n_units_0{0};

  Unit units_1[16];
  int n_units_1{0};

  Unit units_2[16];
  int n_units_2{0};

  Unit units_3[16];
  int n_units_3{0};

  Relic relics[2][24*24];
  int n_relics[2]{0};

  bool out_light[2][16][15 * 15];

  int sap_range_{0};
  int sap_cost_{0};
  int enable_sap_masks_{0};

  bool occupied[2][24][24];

  void Reset(const int sap_range, const int sap_cost, const int enable_sap_masks) {
    n_units_0 = 0;
    n_units_1 = 0;
    n_units_2 = 0;
    n_units_3 = 0;
    sap_range_ = sap_range;
    sap_cost_ = sap_cost;
    n_relics[0] = 0;
    n_relics[1] = 0;
    enable_sap_masks_ = enable_sap_masks;
  }

  void AddUnit_0(const Unit& obj) {
    units_0[n_units_0++] = obj;
  }

  void AddUnit_1(const Unit& obj) {
    units_1[n_units_1++] = obj;
  }

  void AddUnit_2(const Unit& obj) {
    units_2[n_units_2++] = obj;
  }

  void AddUnit_3(const Unit& obj) {
    units_3[n_units_3++] = obj;
  }

  void AddRelic(const Relic& obj) {
    int pi = obj.player_id;
    relics[pi][n_relics[pi]++] = obj;
  }

  void CalcAAM() {
    memset(out_light, false, sizeof(out_light));
    memset(occupied, false, sizeof(occupied));

    for (int i = 0; i < n_relics[0]; ++i) {
      const auto& relic = relics[0][i];
      const int x = relic.x;
      const int y = relic.y;
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          const int to_x = x + j;
          const int to_y = y + k;
          if (to_x < 0 || to_y < 0 || to_x >= 24 || to_y >= 24) {
            continue;
          }
          occupied[0][to_x][to_y] = true;
        }
      }
    }

    for (int i = 0; i < n_units_2; ++i) {
      const auto& unit = units_2[i];
      if (unit.is_fake || unit.energy < 0) {
        continue;
      }
      const int x = unit.x;
      const int y = unit.y;
      const int id = unit.id;
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          const int to_x = x + j;
          const int to_y = y + k;
          if (to_x < 0 || to_y < 0 || to_x >= 24 || to_y >= 24) {
            continue;
          }
          occupied[0][to_x][to_y] = true;
        }
      }
    }

    for (int i = 0; i < n_relics[1]; ++i) {
      const auto& relic = relics[1][i];
      const int x = relic.x;
      const int y = relic.y;
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          const int to_x = x + j;
          const int to_y = y + k;
          if (to_x < 0 || to_y < 0 || to_x >= 24 || to_y >= 24) {
            continue;
          }
          occupied[1][to_x][to_y] = true;
        }
      }
    }

    for (int i = 0; i < n_units_3; ++i) {
      const auto& unit = units_3[i];
      if (unit.is_fake || unit.energy < 0) {
        continue;
      }
      const int x = unit.x;
      const int y = unit.y;
      const int id = unit.id;
      for (int j = -1; j <= 1; ++j) {
        for (int k = -1; k <= 1; ++k) {
          const int to_x = x + j;
          const int to_y = y + k;
          if (to_x < 0 || to_y < 0 || to_x >= 24 || to_y >= 24) {
            continue;
          }
          occupied[1][to_x][to_y] = true;
        }
      }
    }


    for (int i = 0; i < n_units_0; ++i) {
      const auto& unit = units_0[i];
      const int x = unit.x;
      const int y = unit.y;
      const int id = unit.id;

      int action_id = -1;
      for (int j = -7; j <= 7; ++j) {
        for (int k = -7; k <= 7; ++k) {
          action_id++;
          if (std::abs(j) > sap_range_ || std::abs(k) > sap_range_) {
            continue;
          }
          const int to_x = x + j;
          const int to_y = y + k;
          if (to_x < 0 || to_y < 0 || to_x >= 24 || to_y >= 24) {
            continue;
          }
          out_light[0][id][action_id] = !unit.is_fake && unit.energy >= sap_cost_ && (!enable_sap_masks_ || occupied[0][to_x][to_y]);
        }
      }
    }

    for (int i = 0; i < n_units_1; ++i) {
      const auto& unit = units_1[i];
      const int x = unit.x;
      const int y = unit.y;
      const int id = unit.id;

      int action_id = -1;
      for (int j = -7; j <= 7; ++j) {
        for (int k = -7; k <= 7; ++k) {
          action_id++;
          if (std::abs(j) > sap_range_ || std::abs(k) > sap_range_) {
            continue;
          }
          const int to_x = x + j;
          const int to_y = y + k;
          if (to_x < 0 || to_y < 0 || to_x >= 24 || to_y >= 24) {
            continue;
          }
          out_light[1][id][action_id] = !unit.is_fake && unit.energy >= sap_cost_ && (!enable_sap_masks_ || occupied[1][to_x][to_y]);
        }
      }
    }

  }

  void FillAAMForLight(void* ptr_) {
    memcpy(ptr_, out_light, sizeof(out_light));
  }

};

void* CreateMyClass() {
  void* my_class = (void*)(new MyClass());
  return my_class;
}

void DeleteMyClass(void* _my_class_ptr) {
  delete (MyClass*)_my_class_ptr;
}

void Reset(const void* _my_class_ptr, const int sap_range, const int sap_cost, const int enable_sap_masks) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.Reset(sap_range, sap_cost, enable_sap_masks);
}

void AddRelic(const void* _my_class_ptr,
               const int player_id,
               const int x,
               const int y
) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.AddRelic(Relic{player_id, x, y});
}

void AddUnit_0(const void* _my_class_ptr,
             const int id,
             const int x,
             const int y,
             const int energy,
             const int is_fake
) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.AddUnit_0(Unit{id, x, y, energy, is_fake});
}

void AddUnit_1(const void* _my_class_ptr,
               const int id,
               const int x,
               const int y,
               const int energy,
               const int is_fake
) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.AddUnit_1(Unit{id, x, y, energy, is_fake});
}


void AddUnit_2(const void* _my_class_ptr,
               const int id,
               const int x,
               const int y,
               const int energy,
               const int is_fake
) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.AddUnit_2(Unit{id, x, y, energy, is_fake});
}

void AddUnit_3(const void* _my_class_ptr,
               const int id,
               const int x,
               const int y,
               const int energy,
               const int is_fake
) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.AddUnit_3(Unit{id, x, y, energy, is_fake});
}


void CalcAAM(const void* _my_class_ptr) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.CalcAAM();
}

void FillAAMForLight(const void* _my_class_ptr, void* ptr_) {
  MyClass& my_class = *((MyClass*)_my_class_ptr);
  my_class.FillAAMForLight(ptr_);
}


}