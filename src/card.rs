//! Character card normalization, mirroring upstream `src/endpoints/characters.js`:
//! - `getCharaCardV2` → `readFromV2` (spec present) / `convertToV2` (v1)
//! - `charaFormatData` (edit path rebuild)
//! - `humanizedDateTime`
//! - `convertWorldInfoToCharacterBook`
//! - `unsetPrivateFields`
//! - `calculateDataSize` (UTF-16 code-unit semantics)
//!
//! All JSON manipulation uses serde_json with `preserve_order` so field
//! order matches JS object insertion order byte-for-byte in output.

use serde_json::{Map, Value};

use crate::png::TextChunk;

/// Mirror of `humanizedDateTime()`: `YYYY-MM-DD@hhHmmMMmSSsmsmmmms` local time.
pub fn humanized_datetime() -> String {
    let now = chrono::Local::now();
    format!(
        "{}-{}-{}@{}h{}m{}s{}ms",
        now.format("%Y"),
        now.format("%m"),
        now.format("%d"),
        now.format("%H"),
        now.format("%M"),
        now.format("%S"),
        format!("{:03}", now.timestamp_subsec_millis())
    )
}

fn is_object(v: &Value) -> bool {
    v.is_object()
}

/// Mirror of lodash `_.set(obj, path, value)` for dot paths, creating
/// intermediate objects. JS arrays are indexed numerically like lodash.
/// Mirror of lodash `_.set(obj, path, value)` for dot paths, creating
/// intermediate objects. Public for cross-module use (routes/edit).
pub fn set_path_value(obj: &mut Value, path: &str, value: Value) {
    set_path(obj, path, value)
}

fn set_path(obj: &mut Value, path: &str, value: Value) {
    let mut keys: Vec<&str> = path.split('.').collect();
    let last = keys.pop().unwrap();
    let mut cur = obj;
    for key in keys {
        if !cur.is_object() {
            *cur = Value::Object(Map::new());
        }
        cur = cur
            .as_object_mut()
            .unwrap()
            .entry(key.to_string())
            .or_insert(Value::Object(Map::new()));
    }
    if !cur.is_object() {
        *cur = Value::Object(Map::new());
    }
    cur.as_object_mut().unwrap().insert(last.to_string(), value);
}

fn get_path<'a>(obj: &'a Value, path: &str) -> Option<&'a Value> {
    let mut cur = obj;
    for key in path.split('.') {
        cur = cur.get(key)?;
    }
    Some(cur)
}

/// Mirror of `deepMerge` in util.js (Object.assign-based recursive merge).
pub fn deep_merge(target: &Value, source: &Value) -> Value {
    let mut output = match target {
        Value::Object(m) => m.clone(),
        other => return other.clone(),
    };
    if let (Value::Object(t), Value::Object(s)) = (target, source) {
        for (k, sv) in s {
            if is_object(sv) {
                if let Some(tv) = t.get(k) {
                    if is_object(tv) {
                        output.insert(k.clone(), deep_merge(tv, sv));
                        continue;
                    }
                }
                output.insert(k.clone(), sv.clone());
            } else {
                output.insert(k.clone(), sv.clone());
            }
        }
    }
    Value::Object(output)
}

/// Mirror of `readFromV2` (characters.js): for spec-bearing cards, hoist
/// `data.<field>` fields to the top level with defaults and warnings skipped
/// (console output not needed). Sets `chat` default with humanized timestamp.
pub fn read_from_v2(char: &mut Value) {
    if char.get("data").map(|d| !d.is_object()).unwrap_or(true) {
        return; // upstream warns and returns as-is
    }
    // json_data must not propagate (upstream unsets it)
    if let Some(obj) = char.as_object_mut() {
        obj.remove("json_data");
    }

    let field_mappings: &[(&str, &str)] = &[
        ("name", "name"),
        ("description", "description"),
        ("personality", "personality"),
        ("scenario", "scenario"),
        ("first_mes", "first_mes"),
        ("mes_example", "mes_example"),
        ("talkativeness", "extensions.talkativeness"),
        ("fav", "extensions.fav"),
        ("tags", "tags"),
    ];

    let data = char.get("data").cloned().unwrap_or(Value::Null);
    for (char_field, v2_path) in field_mappings {
        let v2_value = data.pointer(&format!("/{}", v2_path.replace('.', "/")));
        if v2_value.is_none() {
            // Backfill defaults for ST extension fields
            let default: Option<Value> = match *v2_path {
                "extensions.talkativeness" => Some(Value::from(0.5)),
                "extensions.fav" => Some(Value::from(false)),
                _ => None,
            };
            if let Some(d) = default {
                if let Some(obj) = char.as_object_mut() {
                    obj.insert(char_field.to_string(), d);
                }
            }
            continue;
        }
        // Upstream warns on mismatch then overwrites: char[charField] = v2Value
        let v = v2_value.unwrap().clone();
        if let Some(obj) = char.as_object_mut() {
            obj.insert(char_field.to_string(), v);
        }
    }

    // chat = char.chat ?? `${char.name} - ${humanizedDateTime()}`
    let has_chat = char.get("chat").map(|c| !c.is_null()).unwrap_or(false);
    if !has_chat {
        let name = char
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("")
            .to_string();
        if let Some(obj) = char.as_object_mut() {
            obj.insert(
                "chat".into(),
                Value::from(format!("{} - {}", name, humanized_datetime())),
            );
        }
    }
}

/// Mirror of `convertToV2`: v1 card → build normalized card via
/// `charaFormatData`-equivalent on the v1 fields.
pub fn convert_to_v2(char: &Value) -> Value {
    let get = |k: &str| char.get(k).cloned().unwrap_or(Value::Null);
    let input = FormatDataInput {
        ch_name: get("name"),
        description: get("description"),
        personality: get("personality"),
        scenario: get("scenario"),
        first_mes: get("first_mes"),
        mes_example: get("mes_example"),
        creator_notes: get("creatorcomment"),
        talkativeness: get("talkativeness"),
        fav: get("fav"),
        creator: get("creator"),
        tags: get("tags"),
        depth_prompt_prompt: get("depth_prompt_prompt"),
        depth_prompt_depth: get("depth_prompt_depth"),
        depth_prompt_role: get("depth_prompt_role"),
        system_prompt: Value::Null,
        post_history_instructions: Value::Null,
        character_version: Value::Null,
        alternate_greetings: Value::Null,
        world: get("world"),
        extensions: Value::Null,
    };
    let mut result = chara_format_data_from_parts(
        &input,
        // json_data: upstream passes JSON.stringify(char) here and re-parses;
        // equivalent to starting from a copy of char without json_data.
        {
            let mut c = char.clone();
            if let Some(o) = c.as_object_mut() {
                o.remove("json_data");
            }
            c
        },
    );
    // result.chat = char.chat ?? default; result.create_date = char.create_date
    let chat = char.get("chat").cloned();
    if let Some(chat) = chat {
        set_path(&mut result, "chat", chat);
    } else {
        let name = char.get("name").and_then(|n| n.as_str()).unwrap_or("");
        set_path(
            &mut result,
            "chat",
            Value::from(format!("{} - {}", name, humanized_datetime())),
        );
    }
    if let Some(cd) = char.get("create_date") {
        set_path(&mut result, "create_date", cd.clone());
    }
    result
}

/// Mirror of `getCharaCardV2` (no hoistDate: /all and /get pass false; /edit
/// equivalent paths handle create_date explicitly).
pub fn get_chara_card_v2(mut json: Value) -> Value {
    let has_spec = json.get("spec").map(|s| !s.is_null()).unwrap_or(false);
    if !has_spec {
        convert_to_v2(&json)
    } else {
        read_from_v2(&mut json);
        json
    }
}

/// Collected inputs for the charaFormatData mirror (edit path).
pub struct FormatDataInput {
    pub ch_name: Value,
    pub description: Value,
    pub personality: Value,
    pub scenario: Value,
    pub first_mes: Value,
    pub mes_example: Value,
    pub creator_notes: Value,
    pub talkativeness: Value,
    pub fav: Value,
    pub creator: Value,
    pub tags: Value,
    pub depth_prompt_prompt: Value,
    pub depth_prompt_depth: Value,
    pub depth_prompt_role: Value,
    pub system_prompt: Value,
    pub post_history_instructions: Value,
    pub character_version: Value,
    pub alternate_greetings: Value,
    pub world: Value,
    pub extensions: Value,
}

fn jstr(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn truthy_fav(v: &Value) -> bool {
    // JS: data.fav == 'true' (loose equality against the string 'true')
    matches!(v, Value::String(s) if s == "true") || v.as_bool() == Some(true)
}

fn tags_value(v: &Value) -> Value {
    match v {
        Value::String(s) => Value::Array(
            s.split(',')
                .map(|x| Value::from(x.trim()))
                .filter(|x| x.as_str().map(|s| !s.is_empty()).unwrap_or(false))
                .collect(),
        ),
        Value::Array(_) => v.clone(),
        _ => Value::Array(vec![]),
    }
}

fn alternate_greetings(v: &Value) -> Value {
    match v {
        Value::Array(_) => v.clone(),
        Value::String(s) => Value::Array(vec![Value::from(s.clone())]),
        _ => Value::Array(vec![]),
    }
}

fn number_or(v: &Value, default: f64) -> f64 {
    // JS: !isNaN(Number(x)) ? Number(x) : default
    match v {
        Value::Number(n) => n.as_f64().unwrap_or(default),
        Value::String(s) => {
            let t = s.trim();
            if t.is_empty() {
                // Number('') is 0 in JS
                0.0
            } else {
                t.parse::<f64>().unwrap_or(default)
            }
        }
        _ => default,
    }
}

/// Mirror of `charaFormatData` without world-info file IO (world book is
/// injected by the caller when `data.world` is set). `existing` is the parsed
/// previous card (JS parses `data.json_data`), preserving unknown/foreign keys.
pub fn chara_format_data_from_parts(input: &FormatDataInput, existing: Value) -> Value {
    let mut char = existing;
    if let Some(o) = char.as_object_mut() {
        o.remove("json_data"); // prevent recursive saving
    }

    // Spec V1 fields
    set_path(&mut char, "name", Value::from(jstr(&input.ch_name)));
    set_path(
        &mut char,
        "description",
        Value::from(or_empty(&input.description)),
    );
    set_path(
        &mut char,
        "personality",
        Value::from(or_empty(&input.personality)),
    );
    set_path(
        &mut char,
        "scenario",
        Value::from(or_empty(&input.scenario)),
    );
    set_path(
        &mut char,
        "first_mes",
        Value::from(or_empty(&input.first_mes)),
    );
    set_path(
        &mut char,
        "mes_example",
        Value::from(or_empty(&input.mes_example)),
    );

    // Old ST extension fields
    set_path(
        &mut char,
        "creatorcomment",
        Value::from(or_empty(&input.creator_notes)),
    );
    set_path(&mut char, "avatar", Value::from("none"));
    set_path(
        &mut char,
        "chat",
        Value::from(format!(
            "{} - {}",
            jstr(&input.ch_name),
            humanized_datetime()
        )),
    );
    let talk = number_or(&input.talkativeness, 0.5);
    let talk = if talk == 0.0 && input.talkativeness.is_null() {
        0.5
    } else {
        talk
    };
    set_path(&mut char, "talkativeness", serde_json::json!(talk));
    set_path(&mut char, "fav", Value::from(truthy_fav(&input.fav)));
    set_path(&mut char, "tags", tags_value(&input.tags));

    // Spec V2 fields
    set_path(&mut char, "spec", Value::from("chara_card_v2"));
    set_path(&mut char, "spec_version", Value::from("2.0"));
    set_path(&mut char, "data.name", Value::from(jstr(&input.ch_name)));
    set_path(
        &mut char,
        "data.description",
        Value::from(or_empty(&input.description)),
    );
    set_path(
        &mut char,
        "data.personality",
        Value::from(or_empty(&input.personality)),
    );
    set_path(
        &mut char,
        "data.scenario",
        Value::from(or_empty(&input.scenario)),
    );
    set_path(
        &mut char,
        "data.first_mes",
        Value::from(or_empty(&input.first_mes)),
    );
    set_path(
        &mut char,
        "data.mes_example",
        Value::from(or_empty(&input.mes_example)),
    );

    // New V2 fields
    set_path(
        &mut char,
        "data.creator_notes",
        Value::from(or_empty(&input.creator_notes)),
    );
    set_path(
        &mut char,
        "data.system_prompt",
        Value::from(or_empty(&input.system_prompt)),
    );
    set_path(
        &mut char,
        "data.post_history_instructions",
        Value::from(or_empty(&input.post_history_instructions)),
    );
    set_path(&mut char, "data.tags", tags_value(&input.tags));
    set_path(
        &mut char,
        "data.creator",
        Value::from(or_empty(&input.creator)),
    );
    set_path(
        &mut char,
        "data.character_version",
        Value::from(or_empty(&input.character_version)),
    );
    set_path(
        &mut char,
        "data.alternate_greetings",
        alternate_greetings(&input.alternate_greetings),
    );

    // ST extension fields
    set_path(
        &mut char,
        "data.extensions.talkativeness",
        serde_json::json!(talk),
    );
    set_path(
        &mut char,
        "data.extensions.fav",
        Value::from(truthy_fav(&input.fav)),
    );
    set_path(
        &mut char,
        "data.extensions.world",
        Value::from(or_empty(&input.world)),
    );

    // Depth prompt
    let depth = number_or(&input.depth_prompt_depth, 4.0);
    set_path(
        &mut char,
        "data.extensions.depth_prompt.prompt",
        input.depth_prompt_prompt.clone(),
    );
    set_path(
        &mut char,
        "data.extensions.depth_prompt.depth",
        serde_json::json!(depth),
    );
    let role = match &input.depth_prompt_role {
        Value::String(s) if !s.is_empty() => Value::from(s.clone()),
        Value::String(_) => Value::from("system"),
        Value::Null => Value::from("system"),
        other => other.clone(),
    };
    set_path(&mut char, "data.extensions.depth_prompt.role", role);

    // data.extensions deep-merge with parsed extensions JSON
    if let Value::String(ext_str) = &input.extensions {
        match serde_json::from_str::<Value>(ext_str) {
            Ok(ext) if ext.is_object() => {
                let base = get_path(&char, "data.extensions")
                    .cloned()
                    .unwrap_or(Value::Object(Map::new()));
                set_path(&mut char, "data.extensions", deep_merge(&base, &ext));
            }
            _ => { /* upstream warns and skips */ }
        }
    }

    char
}

fn or_empty(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        _ => String::new(),
    }
}

/// Mirror of `convertWorldInfoToCharacterBook` (characters.js): converts a
/// world info file's `entries` map into the character_book format.
pub fn convert_world_info_to_character_book(name: &str, entries: &Value) -> Value {
    let mut result = serde_json::json!({ "entries": [], "name": name });
    if let Value::Object(map) = entries {
        let arr = result
            .as_object_mut()
            .unwrap()
            .get_mut("entries")
            .unwrap()
            .as_array_mut()
            .unwrap();
        // JS `for (const index in entries)` iterates own enumerable keys in
        // insertion order; serde_json preserve_order matches.
        let mut keys: Vec<&String> = map.keys().collect();
        // JS iterates integer-like keys in ascending numeric order first.
        keys.sort_by_key(|k| {
            k.parse::<u64>()
                .map(|n| (0u8, n))
                .unwrap_or((1u8, u64::MAX))
        });
        for key in keys {
            let entry = &map[key];
            let get = |k: &str| entry.get(k).cloned().unwrap_or(Value::Null);
            let position = get("position");
            let pos_val = position.as_i64().unwrap_or(0);
            let num_or = |v: Value, default: Value| if v.is_null() { default } else { v };

            let original_entry = serde_json::json!({
                "id": get("uid"),
                "keys": get("key"),
                "secondary_keys": get("keysecondary"),
                "comment": get("comment"),
                "content": get("content"),
                "constant": get("constant"),
                "selective": get("selective"),
                "insertion_order": get("order"),
                "enabled": !get("disable").as_bool().unwrap_or(false),
                "position": if pos_val == 0 { "before_char" } else { "after_char" },
                "use_regex": true,
                "extensions": {
                    "position": position,
                    "exclude_recursion": get("excludeRecursion"),
                    "display_index": get("displayIndex"),
                    "probability": num_or(get("probability"), Value::Null),
                    "useProbability": num_or(get("useProbability"), serde_json::json!(false)),
                    "depth": num_or(get("depth"), serde_json::json!(4)),
                    "selectiveLogic": num_or(get("selectiveLogic"), serde_json::json!(0)),
                    "outlet_name": num_or(get("outletName"), Value::from("")),
                    "group": num_or(get("group"), Value::from("")),
                    "group_override": num_or(get("groupOverride"), serde_json::json!(false)),
                    "group_weight": num_or(get("groupWeight"), Value::Null),
                    "prevent_recursion": get("preventRecursion"),
                    "delay_until_recursion": get("delayUntilRecursion"),
                    "scan_depth": num_or(get("scanDepth"), Value::Null),
                    "match_whole_words": num_or(get("matchWholeWords"), Value::Null),
                    "use_group_scoring": num_or(get("useGroupScoring"), serde_json::json!(false)),
                    "case_sensitive": num_or(get("caseSensitive"), Value::Null),
                    "automation_id": num_or(get("automationId"), Value::from("")),
                    "role": num_or(get("role"), serde_json::json!(0)),
                    "vectorized": num_or(get("vectorized"), serde_json::json!(false)),
                    "sticky": num_or(get("sticky"), Value::Null),
                    "cooldown": num_or(get("cooldown"), Value::Null),
                    "delay": num_or(get("delay"), Value::Null),
                    "match_persona_description": get("matchPersonaDescription"),
                    "match_character_description": get("matchCharacterDescription"),
                    "match_character_personality": get("matchCharacterPersonality"),
                    "match_character_depth_prompt": get("matchCharacterDepthPrompt"),
                    "match_scenario": get("matchScenario"),
                    "match_creator_notes": get("matchCreatorNotes"),
                    "triggers": num_or(get("triggers"), Value::Array(vec![])),
                    "ignore_budget": get("ignoreBudget"),
                },
            });
            // Spread entry.extensions first (JS: ...entry.extensions), then
            // explicit keys override — handled above by always setting keys,
            // but any extra keys in entry.extensions must be preserved too.
            let mut ext = original_entry["extensions"].as_object().cloned().unwrap();
            if let Some(src_ext) = entry.get("extensions").and_then(|e| e.as_object()) {
                for (k, v) in src_ext {
                    ext.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
            let mut oe = original_entry;
            oe["extensions"] = Value::Object(ext);
            arr.push(oe);
        }
    }
    result
}

/// Mirror of `toShallow` (characters.js): fields used to display the
/// character list. lodash `_.get(path, default)` semantics: missing/null
/// paths fall back to the default.
pub fn to_shallow(character: &Value) -> Value {
    let get = |path: &str, default: Value| -> Value {
        match get_path(character, path) {
            Some(v) if !v.is_null() => v.clone(),
            _ => default,
        }
    };
    let get_str = |path: &str, default: &str| -> Value {
        match get_path(character, path) {
            Some(v) if !v.is_null() => v.clone(),
            _ => Value::from(default),
        }
    };
    serde_json::json!({
        "shallow": true,
        "name": character.get("name").cloned().unwrap_or(Value::Null),
        "avatar": character.get("avatar").cloned().unwrap_or(Value::Null),
        "chat": character.get("chat").cloned().unwrap_or(Value::Null),
        "fav": character.get("fav").cloned().unwrap_or(Value::Null),
        "date_added": character.get("date_added").cloned().unwrap_or(Value::Null),
        "create_date": character.get("create_date").cloned().unwrap_or(Value::Null),
        "date_last_chat": character.get("date_last_chat").cloned().unwrap_or(Value::Null),
        "chat_size": character.get("chat_size").cloned().unwrap_or(Value::Null),
        "data_size": character.get("data_size").cloned().unwrap_or(Value::Null),
        "tags": character.get("tags").cloned().unwrap_or(Value::Null),
        "data": {
            "name": get_str("data.name", ""),
            "character_version": get_str("data.character_version", ""),
            "creator": get_str("data.creator", ""),
            "creator_notes": get_str("data.creator_notes", ""),
            "tags": get("data.tags", Value::Array(vec![])),
            "extensions": {
                "fav": get("data.extensions.fav", Value::from(false)),
                "world": get_str("data.extensions.world", ""),
            },
        },
    })
}

/// Mirror of `unsetPrivateFields` (used by the upcoming /export json route;
/// kept for phase B).
#[allow(dead_code)]
pub fn unset_private_fields(char: &mut Value) {
    set_path(char, "fav", Value::from(false));
    set_path(char, "data.extensions.fav", Value::from(false));
    if let Some(o) = char.as_object_mut() {
        o.remove("chat");
    }
}

/// Mirror of `calculateDataSize`: sum of `String(val).length` over
/// `Object.values(data)` — UTF-16 code units, not bytes or chars.
pub fn calculate_data_size(data: &Value) -> usize {
    match data {
        Value::Object(map) => map
            .values()
            .map(|v| match v {
                Value::String(s) => s.encode_utf16().count(),
                Value::Bool(b) => b.to_string().len(),
                Value::Number(n) => n.to_string().len(),
                Value::Null => 4, // String(null).length
                other => other.to_string().encode_utf16().count(),
            })
            .sum(),
        _ => 0,
    }
}

/// Full single-card processing, mirroring `processCharacter` with
/// `shallow: false`. Returns the response object (never fails: mirrors the
/// upstream catch that returns a zeroed stub).
pub struct CardDeps {
    pub characters_dir: std::path::PathBuf,
    pub chats_dir: std::path::PathBuf,
    pub worlds_dir: std::path::PathBuf,
}

impl Clone for CardDeps {
    fn clone(&self) -> Self {
        Self {
            characters_dir: self.characters_dir.clone(),
            chats_dir: self.chats_dir.clone(),
            worlds_dir: self.worlds_dir.clone(),
        }
    }
}

pub fn process_character(item: &str, deps: &CardDeps) -> Value {
    let file_path = deps.characters_dir.join(item);
    match process_character_inner(item, &file_path, deps) {
        Ok(v) => v,
        Err(e) => {
            tracing::error!("Could not process character: {item} ({e})");
            serde_json::json!({
                "date_added": 0,
                "date_last_chat": 0,
                "chat_size": 0,
            })
        }
    }
}

fn process_character_inner(
    item: &str,
    file_path: &std::path::Path,
    deps: &CardDeps,
) -> Result<Value, String> {
    let chunks: Vec<TextChunk> = crate::png::read_text_chunks(file_path)
        .map_err(|e| format!("failed to read PNG chunks: {e}"))?;
    if chunks.is_empty() {
        return Err("PNG metadata does not contain any text chunks".into());
    }
    let img_data = crate::png::extract_card_json(&chunks)?;
    let parsed: Value =
        serde_json::from_str(&img_data).map_err(|e| format!("invalid card JSON: {e}"))?;
    let value = process_character_from_card(item, parsed, deps, file_path)?;
    Ok(with_json_data(value, &img_data))
}

/// Assemble the response object from an already-parsed card (used by both
/// the full PNG path and the disk-cache path, which supplies the cached card
/// JSON directly and skips chunk scanning).
pub fn process_character_from_card(
    item: &str,
    parsed: Value,
    deps: &CardDeps,
    file_path: &std::path::Path,
) -> Result<Value, String> {
    let mut json_object = get_chara_card_v2(parsed);
    // jsonObject.avatar = item
    if let Some(o) = json_object.as_object_mut() {
        o.insert("avatar".into(), Value::from(item));
    }

    // character.json_data is applied by the caller via with_json_data():
    // the full PNG path holds img_data; the disk-cache path holds the cached
    // raw string. Both must be byte-identical to the original card JSON.

    let stat = std::fs::metadata(file_path).map_err(|e| e.to_string())?;
    let ctime_ms = ctime_millis(&stat);
    let obj = json_object.as_object().expect("object");

    // create_date = jsonObject.create_date || new Date(round(ctime)).toISOString()
    let create_date = match obj.get("create_date") {
        Some(v) if !v.is_null() => v.clone(),
        _ => Value::from(ms_to_iso(ctime_ms)),
    };

    // Chat stats
    let char_dir_name = item.strip_suffix(".png").unwrap_or(item);
    let chats_dir = deps.chats_dir.join(char_dir_name);
    let (chat_size, date_last_chat) = calculate_chat_size(&chats_dir);

    let data_size = obj.get("data").map(calculate_data_size).unwrap_or(0);

    let mut out = json_object;
    let o = out.as_object_mut().unwrap();
    o.insert("date_added".into(), serde_json::json!(ctime_ms));
    o.insert("create_date".into(), create_date);
    o.insert("date_last_chat".into(), serde_json::json!(date_last_chat));
    o.insert("chat_size".into(), serde_json::json!(chat_size));
    o.insert("data_size".into(), serde_json::json!(data_size));
    Ok(out)
}

/// Restore the verbatim `json_data` raw string after assembly (the disk
/// cache stores exactly this string, so the cached path preserves it too).
pub fn with_json_data(mut value: Value, raw: &str) -> Value {
    if let Some(o) = value.as_object_mut() {
        o.insert("json_data".into(), Value::from(raw.to_string()));
    }
    value
}

/// Mirror of `calculateChatSize`.
fn calculate_chat_size(char_dir: &std::path::Path) -> (u64, f64) {
    let mut chat_size: u64 = 0;
    let mut date_last_chat: f64 = 0.0;
    if let Ok(entries) = std::fs::read_dir(char_dir) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    chat_size += meta.len();
                    let mtime = meta
                        .modified()
                        .ok()
                        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                        .map(|d| d.as_millis() as f64)
                        .unwrap_or(0.0);
                    date_last_chat = date_last_chat.max(mtime);
                }
            }
        }
    }
    (chat_size, date_last_chat)
}

/// ctimeMs semantics: Windows → creation time; Unix → inode change time
/// (Node's statSync().ctimeMs). Falls back to mtime when unavailable.
fn ctime_millis(meta: &std::fs::Metadata) -> f64 {
    #[cfg(windows)]
    {
        meta.created()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_millis() as f64)
            .unwrap_or(0.0)
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let ctime_secs = meta.ctime();
        let ctime_nanos = meta.ctime_nsec();
        (ctime_secs as f64 * 1000.0) + (ctime_nanos as f64 / 1_000_000.0)
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = meta;
        0.0
    }
}

/// Mirror of JS `new Date(Math.round(ms)).toISOString()`.
fn ms_to_iso(ms: f64) -> String {
    let secs = (ms / 1000.0).floor() as i64;
    let millis = (ms - secs as f64 * 1000.0).round() as u32;
    chrono::DateTime::from_timestamp(secs, millis * 1_000_000)
        .map(|dt| dt.to_rfc3339_opts(chrono::SecondsFormat::Millis, true))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanized_format() {
        let s = humanized_datetime();
        // YYYY-MM-DD@HHhMMmSSsmsNNNms
        let re = regex_like_check(&s);
        assert!(re, "got {s}");
    }

    fn regex_like_check(s: &str) -> bool {
        let b = s.as_bytes();
        // 10 digits + '-' + '@' layout check
        b.len() >= 21
            && b[4] == b'-'
            && b[7] == b'-'
            && b[10] == b'@'
            && b[13] == b'h'
            && b[16] == b'm'
            && b[19] == b's'
            && s.ends_with("ms")
    }

    #[test]
    fn utf16_data_size() {
        let data = serde_json::json!({"a": "é", "b": "😀"});
        // "é" = 1 utf16 unit; "😀" = 2 utf16 units; total 3
        assert_eq!(calculate_data_size(&data), 3);
    }

    #[test]
    fn read_from_v2_defaults() {
        let mut card = serde_json::json!({
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": { "name": "Test", "description": "d" }
        });
        read_from_v2(&mut card);
        assert_eq!(card["talkativeness"], 0.5);
        assert_eq!(card["fav"], false);
        assert!(card["chat"].as_str().unwrap().contains("Test - "));
        assert!(card.get("json_data").is_none());
    }

    #[test]
    fn world_book_conversion() {
        let entries = serde_json::json!({
            "0": { "uid": 0, "key": ["a"], "keysecondary": [], "content": "x",
                    "constant": false, "selective": true, "order": 100,
                    "position": 0, "disable": false, "displayIndex": 0,
                    "extensions": { "custom": 1 } }
        });
        let book = convert_world_info_to_character_book("w", &entries);
        assert_eq!(book["name"], "w");
        assert_eq!(book["entries"][0]["position"], "before_char");
        assert_eq!(book["entries"][0]["enabled"], true);
        assert_eq!(book["entries"][0]["extensions"]["custom"], 1);
        assert_eq!(
            book["entries"][0]["extensions"]["probability"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn chara_format_data_preserves_foreign_keys() {
        let input = FormatDataInput {
            ch_name: Value::from("NewName"),
            description: Value::from("desc"),
            personality: Value::Null,
            scenario: Value::Null,
            first_mes: Value::Null,
            mes_example: Value::Null,
            creator_notes: Value::Null,
            talkativeness: Value::Null,
            fav: Value::Null,
            creator: Value::Null,
            tags: Value::Null,
            depth_prompt_prompt: Value::Null,
            depth_prompt_depth: Value::Null,
            depth_prompt_role: Value::Null,
            system_prompt: Value::Null,
            post_history_instructions: Value::Null,
            character_version: Value::Null,
            alternate_greetings: Value::Null,
            world: Value::Null,
            extensions: Value::Null,
        };
        let existing = serde_json::json!({"custom_field": "keep me", "name": "Old"});
        let out = chara_format_data_from_parts(&input, existing);
        assert_eq!(out["custom_field"], "keep me");
        assert_eq!(out["name"], "NewName");
        assert_eq!(out["spec"], "chara_card_v2");
        assert_eq!(out["data"]["extensions"]["talkativeness"], 0.5);
    }

    #[test]
    fn to_shallow_matches_upstream_toshallow() {
        // Full character → shallow projection with exact field set
        let character = serde_json::json!({
            "name": "T", "avatar": "t.png", "chat": "T - chat",
            "fav": true, "date_added": 1.0, "create_date": "2024-01-01",
            "date_last_chat": 2.0, "chat_size": 3, "data_size": 4,
            "tags": ["x"],
            "json_data": "{\"huge\": true}",
            "description": "stripped",
            "first_mes": "stripped",
            "data": {
                "name": "T", "description": "stripped",
                "character_version": "1.0", "creator": "c",
                "creator_notes": "n", "tags": ["x"],
                "extensions": {"fav": true, "world": "w"},
                "character_book": {"entries": []}
            }
        });
        let s = to_shallow(&character);
        // Exact key set (no json_data/description/first_mes/book)
        let keys: Vec<&str> = s.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "shallow",
                "name",
                "avatar",
                "chat",
                "fav",
                "date_added",
                "create_date",
                "date_last_chat",
                "chat_size",
                "data_size",
                "tags",
                "data"
            ]
        );
        assert_eq!(s["shallow"], true);
        assert_eq!(s["data"]["name"], "T");
        assert_eq!(s["data"]["extensions"]["world"], "w");
        assert!(s.get("data").unwrap().get("description").is_none());
        assert!(s.get("json_data").is_none());
    }

    #[test]
    fn to_shallow_defaults_match_lodash_get() {
        // Missing paths fall back to defaults like _.get(path, default)
        let character = serde_json::json!({ "name": "Min" });
        let s = to_shallow(&character);
        assert_eq!(s["shallow"], true);
        assert_eq!(s["name"], "Min");
        assert_eq!(s["data"]["name"], "");
        assert_eq!(s["data"]["character_version"], "");
        assert_eq!(s["data"]["creator"], "");
        assert_eq!(s["data"]["creator_notes"], "");
        assert_eq!(s["data"]["tags"], serde_json::Value::Array(vec![]));
        assert_eq!(s["data"]["extensions"]["fav"], false);
        assert_eq!(s["data"]["extensions"]["world"], "");
        // Top-level missing fields pass through as null (JS undefined behavior:
        // JSON.stringify drops undefined, but here the key set must match the
        // upstream object literal, which always contains these keys)
        assert!(s.get("avatar").is_some());
    }
}
