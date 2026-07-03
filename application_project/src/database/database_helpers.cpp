#include "database/database_helpers.h"

#include <iostream>
#include <sqlite3.h>

#include "database/database_map.h"
#include "database/param_info.h"
#include "database/query_template_manager.h"

int prepare_stmt(sqlite3 *database, sqlite3_stmt *&stmt, const char *query,  const bool verbose)
{
    int ret = sqlite3_prepare_v2(database, query, -1, &stmt, NULL);
    if (ret != SQLITE_OK)
    {
        if (verbose) { std::cout << "could not prepare statement: " << sqlite3_errmsg(database) << std::endl; }
        (void) free_stmt(database, stmt, verbose);
        return ret;
    }
    return DB_OK;
}

int free_stmt(sqlite3 *database, sqlite3_stmt *stmt, const bool verbose)
{
    int ret = sqlite3_finalize(stmt);
    stmt = nullptr;
    if (ret != SQLITE_OK)
    {
        if (verbose) { std::cout << "could not free the query: " << sqlite3_errmsg(database) << std::endl; }
        return ret;
    }
    return DB_OK;
}

int bind_stmt(sqlite3 *database, sqlite3_stmt *stmt, QueryTypes queryType, const ParamInfo &paramInfo, 
    int &paramIdx, const bool verbose)
{
    if (queryType != QueryTypes::INSERT)
    {
        return DB_SKIP;
    }
    int ret = bind_values(stmt, queryType, paramInfo, paramIdx); // fn auto increments the paramIdx
    if (ret != SQLITE_OK)
    {
        if (verbose) { std::cout << "could not bind values: " << sqlite3_errmsg(database) << std::endl; }
        (void) free_stmt(database, stmt, verbose);
        return ret;
    }
    return DB_OK;
}

int step_stmt(sqlite3 *database, sqlite3_stmt *stmt, const size_t maxCols, const QueryTemplateManager &queryTemplateManager,
    const std::string &tableName, const TableConstructor &tableInfo, const bool verbose)
{
    const int nCols = sqlite3_column_count(stmt);
    int ret = 0;
    while (true) // todo: hard cap based on max num of rows
    {
        ret = sqlite3_step(stmt);
        switch (ret)
        {
            case SQLITE_ROW:
            {
                for (int i = 0; i < nCols; ++i)
                {
                    const char *colName = sqlite3_column_name(stmt, i);
                    const unsigned char *colText = sqlite3_column_text(stmt, i);
                    ret = queryTemplateManager.validate_templates(colName, colText, tableName, tableInfo); // this gets skipped if the database is empty
                    if (ret != DB_OK)
                    {
                        return ret;
                    }
                }
                break;
            }
            case SQLITE_DONE:
            {
                return DB_OK;
            }
            default:
            {

                if (verbose) { std::cout << "could not execute the query: " << sqlite3_errmsg(database) << std::endl; }
                (void) free_stmt(database, stmt, verbose);
                return DB_ERR_UNKNOWN;
            }
        }
    }
    return DB_OK;
}