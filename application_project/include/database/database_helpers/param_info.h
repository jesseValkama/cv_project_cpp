#ifndef PARAMINFO_H
#define PARAMINFO_H

#include <optional>
#include <string>
#include <variant>

enum SqliteTypes
{
	TYPE_INT = 1,
	TYPE_REAL = 2,
	TYPE_TEXT = 3,
	TYPE_BLOB = 4,
	TYPE_NULL = 5
};

enum Queries
{
	CREATE = 0,
	INSERT = 1,
	UPDATE = 2
};

struct ForeignKey
{
	using QueryVariant = std::variant<const char *, int, double>;

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

#endif // PARAMINFO_H