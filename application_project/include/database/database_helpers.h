#ifndef DATABASE_HELPERS
#define DATABASE_HELPERS

#include <sqlite3.h>

#include "database/param_info.h"
#include "database/database_map.h"
#include "database/query_template_manager.h"

int prepare_stmt(sqlite3 *database, sqlite3_stmt *&stmt, const char *query, const bool verbose = true);
/*
*/

int free_stmt(sqlite3 *database, sqlite3_stmt *stmt, const bool verbose = true);
/*
*/

int bind_stmt(sqlite3 *database, sqlite3_stmt *stmt, QueryTypes queryType, const ParamInfo &paramInfo, 
    int &paramIdx, const bool verbose = true);
/*
*/

int step_stmt(sqlite3 *database, sqlite3_stmt *stmt, const size_t maxCols, const QueryTemplateManager &queryTemplateManager,
     const std::string &tableName, const TableConstructor &tableInfo, const bool verbose = true);
/*
*/

#endif // DATABASE_HELPERS