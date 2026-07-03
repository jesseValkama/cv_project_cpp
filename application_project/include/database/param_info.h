#ifndef PARAMINFO_H
#define PARAMINFO_H

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>

enum DBRets
{
	DB_OK = 0,
	DB_SKIP = 1,
	DB_ERR_UNKNOWN = 10,
	DB_ERR_SCHEMA = 11
};

enum SqliteTypes
{
	TYPE_INT = 1,
	TYPE_REAL = 2,
	TYPE_TEXT = 3,
	TYPE_BLOB = 4,
	TYPE_NULL = 5
};

enum QueryTypes
{
	CREATE = 0,
	INSERT = 1,
	UPDATE = 2,
	DELETE = 3,
	VALIDATE = 4
};

struct ForeignKey
{
	std::string referenceTable = "";
	std::string referenceColumn = "";
	bool cascade = true;

	ForeignKey(const std::string referenceTable, const std::string referenceColumn, const bool cascade = true)
		: referenceTable(referenceTable), referenceColumn(referenceColumn), cascade(cascade) {};
};

struct ParamInfo
{
	using QueryVariant = std::variant<const char *, int, double>;

	QueryVariant value = 0;
	SqliteTypes type = TYPE_NULL;
	bool notNull = false;
	bool unique = false;
	bool autoIncrement = false;
	bool primaryKey = false;
	std::optional<ForeignKey> foreignKey = std::nullopt;

	ParamInfo(const QueryVariant value, const SqliteTypes type, const bool notNull = false, const bool unique = false, const bool autoIncrement = false, const bool primaryKey = false, const std::optional<ForeignKey> foreignKey = std::nullopt) 
		: value(value), type(type), notNull(notNull), unique(unique), autoIncrement(autoIncrement), primaryKey(primaryKey), foreignKey(foreignKey) {};
};

extern std::unordered_map<SqliteTypes, std::string> sqliteTypesMap;

#endif // PARAMINFO_H