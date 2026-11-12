-- cacography: record each page's layout for the renderer, and swap rendered pages in.
--
-- At shipout, the page box is walked once to find every glyph, rule and line with its
-- position on the page. That scene is written as JSON with a hash. If the renderer has
-- already drawn a page with the same hash, the drawing becomes the page and the typeset
-- content stays as invisible text, so the PDF remains searchable.

cacography = cacography or {}
local M = cacography

local SP_PER_PT = 65536
local RUNNING = -1073741824

local GLYPH, HLIST, VLIST = node.id("glyph"), node.id("hlist"), node.id("vlist")
local RULE, GLUE, KERN = node.id("rule"), node.id("glue"), node.id("kern")
local DISC, MATH = node.id("disc"), node.id("math")
local WHATSIT = node.id("whatsit")
local USER = node.subtype("user_defined")
local PAGE_OPTIONS = 0x63616361 -- marks a \cacographypage setting riding along with its page
local NOTE_ANCHOR, NOTE_START = 0x63616e01, 0x63616e02 -- where a \cacographynote* points, and its note
local NOTE_RIGHT = 0x63616e03 -- a note to move to the right margin
local LITERAL = node.subtype("pdf_literal")
local COLORSTACK = node.subtype("pdf_colorstack")
local SP_PER_BP = 65781.76
local DRAWN = 0x63616403 -- a readable copy of a drawing command, written beside it (see M.literal)
-- Rule subtypes that draw ink; 1 to 3 are box, image and empty.
local INK_RULE = { [0] = true, [4] = true, [5] = true, [6] = true, [7] = true, [8] = true, [9] = true }

M.options = {}
M.page = 0
M.waiting = 0

local function pt(v)
  return math.floor(v / SP_PER_PT * 1000 + 0.5) / 1000
end

local function from_hex_utf16(hex)
  local out, i = {}, 1
  while i + 3 <= #hex do
    local u = tonumber(hex:sub(i, i + 3), 16)
    i = i + 4
    if u >= 0xD800 and u <= 0xDBFF and i + 3 <= #hex then
      local lo = tonumber(hex:sub(i, i + 3), 16)
      i = i + 4
      u = 0x10000 + (u - 0xD800) * 0x400 + (lo - 0xDC00)
    end
    out[#out + 1] = utf8.char(u)
  end
  return table.concat(out)
end

local fonts_seen = {}
local function font_info(id)
  local fi = fonts_seen[id]
  if fi then
    return fi
  end
  local f = font.getfont(id) or {}
  local name = string.lower(f.fullname or f.name or f.psname or "")
  fi = {
    size = (f.size or 0) / SP_PER_PT,
    math = f.mathparameters ~= nil or f.MathConstants ~= nil or name:find("math") ~= nil,
    bold = name:find("bold") ~= nil,
    italic = name:find("italic") ~= nil or name:find("oblique") ~= nil,
    chars = f.characters or {},
    xheight = ((f.parameters or {}).x_height or 0) / SP_PER_PT,
  }
  fonts_seen[id] = fi
  return fi
end

-- The text a glyph stands for. Ligatures and math variants carry their meaning in
-- the font's unicode data, not in the glyph's code point.
local function glyph_text(n, fi)
  local c = fi.chars[n.char]
  if c then
    local u, text = c.unicode, nil
    if type(u) == "number" then
      text = utf8.char(u)
    elseif type(u) == "table" then
      local out = {}
      for _, v in ipairs(u) do
        out[#out + 1] = utf8.char(v)
      end
      text = table.concat(out)
    elseif type(c.tounicode) == "string" then
      text = from_hex_utf16(c.tounicode)
    end
    if text and text ~= "" then
      return text
    end
  end
  if n.char < 0xE000 or (n.char > 0xF8FF and n.char < 0xF0000) then
    return utf8.char(n.char)
  end
  return ""
end

-- Page walk -----------------------------------------------------------------

local scene

local function new_scene()
  return { glyphs = {}, rules = {}, lines = {}, line = 0, word = 0, options = {}, spans = {}, notes = {}, right = {}, colours = {},
           draw = {}, ctm = { 1, 0, 0, 1, 0, 0 }, saved = {}, anchor = nil }
end

-- Drawings (TikZ and anything else that writes PDF drawing commands) are recorded as they
-- are, with where they sit, for the renderer to redraw by hand. Their q, Q and cm are
-- followed too: TikZ places a label's text by moving the coordinate system, not the box,
-- so text inside such a scope is put where it really appears.
local function identity(m)
  return m[1] == 1 and m[2] == 0 and m[3] == 0 and m[4] == 1 and m[5] == 0 and m[6] == 0
end

local function drawing(data, x, y, line)
  local d = scene.draw
  d[#d + 1] = { x = pt(x), y = pt(y), d = data, line = line or 0 }
  local nums = {}
  for tok in string.gmatch(data, "%S+") do
    local v = tonumber(tok)
    if v then
      nums[#nums + 1] = v
    else
      if tok == "q" then
        scene.saved[#scene.saved + 1] = { table.unpack(scene.ctm) }
      elseif tok == "Q" then
        scene.ctm = table.remove(scene.saved) or { 1, 0, 0, 1, 0, 0 }
      elseif tok == "cm" and #nums >= 6 then
        local a, b, c, dd, e, f = table.unpack(nums, #nums - 5, #nums)
        local m = scene.ctm
        if identity(m) then
          scene.anchor = { x, y }
        end
        scene.ctm = { a * m[1] + b * m[3], a * m[2] + b * m[4], c * m[1] + dd * m[3], c * m[2] + dd * m[4],
                      e * m[1] + f * m[3] + m[5], e * m[2] + f * m[4] + m[6] }
      end
      nums = {}
    end
  end
end

-- Where a point really is, under the current drawing transform.
local function placed(x, y)
  local m, a = scene.ctm, scene.anchor
  if identity(m) or not a then
    return x, y
  end
  local dx, dy = (x - a[1]) / SP_PER_BP, (a[2] - y) / SP_PER_BP
  return a[1] + (m[1] * dx + m[3] * dy + m[5]) * SP_PER_BP, a[2] - (m[2] * dx + m[4] * dy + m[6]) * SP_PER_BP
end

-- Markers riding in the text: a page's own settings, and margin notes' arrows.
local function page_options(n, x, y, line)
  if n.id == WHATSIT and n.subtype == COLORSTACK and n.stack == 0 then
    -- Colours (xcolor's, so TikZ's fills and grey grid lines) go on a colour stack, not
    -- through pgf's literals: record each change as a drawing command, so the renderer
    -- knows how pale a line or how dark a fill is.
    local c = scene.colours
    if n.command == 1 then
      c[#c + 1] = n.data
    elseif n.command == 0 then
      c[math.max(#c, 1)] = n.data
    elseif n.command == 2 then
      c[#c] = nil
    end
    local now = (n.command == 2) and (c[#c] or "0 g 0 G") or n.data
    if now and (now:find(" g") or now:find(" rg") or now:find(" k")) then
      local d = scene.draw
      d[#d + 1] = { x = pt(x), y = pt(y), d = now, line = line or 0 }
    end
    return
  end
  if n.id ~= WHATSIT or n.subtype ~= USER then
    return
  end
  if n.user_id == DRAWN then
    drawing(n.value, x, y, line)
    return
  end
  if n.user_id == PAGE_OPTIONS then
    local o = scene.options
    o[#o + 1] = n.value
  elseif n.user_id == NOTE_ANCHOR or n.user_id == NOTE_START then
    local note = scene.notes[n.value] or { id = n.value }
    scene.notes[n.value] = note
    if n.user_id == NOTE_ANCHOR then
      note.x, note.y, note.line = pt(x), pt(y), line or 0
    else
      note.from = line or 0
    end
  elseif n.user_id == NOTE_RIGHT then
    scene.right[#scene.right + 1] = line or 0
  end
end

local walk_v

local function walk_h(head, parent, x, y, line)
  for n in node.traverse(head) do
    local id = n.id
    if id == GLYPH then
      local fi = font_info(n.font)
      local g = scene.glyphs
      local gx, gy = placed(x + n.xoffset, y - n.yoffset)
      local m = scene.ctm
      -- Inside a drawing (a TikZ label, an axis's ticks), TeX's position is where the
      -- label belongs; a turned coordinate system (a rotated label) turns its text too.
      local fig = #scene.saved > 0 or not identity(m)
      local rot = fig and math.atan(m[2], m[1]) or 0
      g[#g + 1] = {
        x = pt(gx), y = pt(gy), w = pt(n.width), h = pt(n.height), d = pt(n.depth),
        t = glyph_text(n, fi), size = fi.size, xh = fi.xheight,
        math = fi.math, bold = fi.bold, italic = fi.italic, line = line, word = scene.word,
        v = node.has_attribute(n, M.variant),
        cross = node.has_attribute(n, M.crossout),
        erased = node.has_attribute(n, M.erased_attr),
        fig = fig or nil, rot = math.abs(rot) > 0.01 and rot or nil,
      }
      x = x + n.width
    elseif id == HLIST then
      if n.list then
        walk_h(n.list, n, x, y + n.shift, line)
      end
      x = x + n.width
    elseif id == VLIST then
      if n.list then
        -- A box wholly beside its line (a margin note) holds lines of its own.
        local span = line and scene.spans[line]
        local beside = span and (x + n.width <= span[1] + SP_PER_PT or x >= span[2] - SP_PER_PT)
        walk_v(n.list, n, x, y + n.shift - n.height, (not beside) and line or nil)
      end
      x = x + n.width
    elseif id == RULE then
      local h = n.height == RUNNING and parent.height or n.height
      local d = n.depth == RUNNING and parent.depth or n.depth
      if INK_RULE[n.subtype] and n.width > 0 and h + d > 0 then
        local r = scene.rules
        r[#r + 1] = { x = pt(x), y = pt(y - h), w = pt(n.width), h = pt(h + d), line = line }
      end
      x = x + n.width
    elseif id == GLUE then
      x = x + node.effective_glue(n, parent)
      scene.word = scene.word + 1
    elseif id == KERN then
      x = x + n.kern
    elseif id == DISC then
      if n.replace then
        for m in node.traverse(n.replace) do
          if m.id == GLYPH or m.id == KERN or m.id == HLIST then
            -- Walk the unbroken form one node at a time to advance x correctly.
            local saved = m.next
            m.next = nil
            walk_h(m, parent, x, y, line)
            m.next = saved
            x = x + (m.id == KERN and m.kern or m.width)
          end
        end
      end
    elseif id == MATH then
      x = x + (node.effective_glue(n, parent) or ((n.surround or 0) + (n.width or 0)))
    else
      page_options(n, x, y, line)
    end
  end
end

function walk_v(head, parent, x, y, line)
  for n in node.traverse(head) do
    local id = n.id
    if id == HLIST then
      local base = y + n.height
      local this = line
      if not this then
        -- A line of the page. Boxes nested inside it (fractions, scripts) stay part of it.
        scene.line = scene.line + 1
        scene.word = scene.word + 1
        this = scene.line
        local l = scene.lines
        l[#l + 1] = { id = this, x = pt(x + n.shift), y = pt(base), w = pt(n.width), h = pt(n.height), d = pt(n.depth) }
        scene.spans[this] = { x + n.shift, x + n.shift + n.width }
      end
      if n.list then
        walk_h(n.list, n, x + n.shift, base, this)
      end
      y = base + n.depth
    elseif id == VLIST then
      if n.list then
        walk_v(n.list, n, x + n.shift, y, line)
      end
      y = y + n.height + n.depth
    elseif id == RULE then
      local w = n.width == RUNNING and parent.width or n.width
      if INK_RULE[n.subtype] and w > 0 and n.height + n.depth > 0 then
        local r = scene.rules
        r[#r + 1] = { x = pt(x), y = pt(y), w = pt(w), h = pt(n.height + n.depth), line = line or 0 }
      end
      y = y + n.height + n.depth
    elseif id == GLUE then
      y = y + node.effective_glue(n, parent)
    elseif id == KERN then
      y = y + n.kern
    else
      page_options(n, x, y, line)
    end
  end
end

-- JSON ----------------------------------------------------------------------

local function json(v)
  local t = type(v)
  if t == "table" then
    if #v > 0 or next(v) == nil then
      local out = {}
      for i = 1, #v do
        out[i] = json(v[i])
      end
      return "[" .. table.concat(out, ",") .. "]"
    end
    local keys = {}
    for k in pairs(v) do
      keys[#keys + 1] = k
    end
    table.sort(keys)
    local out = {}
    for _, k in ipairs(keys) do
      out[#out + 1] = json(tostring(k)) .. ":" .. json(v[k])
    end
    return "{" .. table.concat(out, ",") .. "}"
  elseif t == "string" then
    return '"' .. v:gsub('[%c"\\]', function(c)
      return string.format("\\u%04x", string.byte(c))
    end) .. '"'
  elseif t == "number" then
    if v == math.floor(v) then
      return string.format("%d", v)
    end
    return string.format("%.3f", v)
  elseif t == "boolean" then
    return v and "true" or "false"
  end
  return "null"
end

local function read_file(path)
  local f = io.open(path, "rb")
  if not f then
    return nil
  end
  local s = f:read("*a")
  f:close()
  return s
end

local function write_file(path, s)
  local f = assert(io.open(path, "wb"))
  f:write(s)
  f:close()
end

-- Swapping in the drawn page ------------------------------------------------

local function origin()
  local ok, h, v = pcall(pdf.getorigin)
  if not ok or not h then
    h, v = 4736286, 4736286 -- one inch
  end
  return h + tex.hoffset, v + tex.voffset
end

local function hide_ink(head)
  for n in node.traverse(head) do
    if n.id == RULE and INK_RULE[n.subtype] then
      n.height, n.depth = 0, 0
    elseif n.id == WHATSIT and n.subtype == LITERAL and n.next and n.next.id == WHATSIT
        and n.next.subtype == USER and n.next.user_id == DRAWN then
      -- Drawn by hand now: keep only what changes the coordinates, so text stays in place.
      local keep = {}
      local nums = {}
      for tok in string.gmatch(n.next.value, "%S+") do
        if tonumber(tok) then
          nums[#nums + 1] = tok
        else
          if tok == "q" or tok == "Q" then
            keep[#keep + 1] = tok
          elseif tok == "cm" then
            keep[#keep + 1] = table.concat(nums, " ", math.max(1, #nums - 5)) .. " cm"
          end
          nums = {}
        end
      end
      node.direct.setdata(node.direct.todirect(n), table.concat(keep, " "))
    elseif (n.id == HLIST or n.id == VLIST) and n.list then
      hide_ink(n.list)
    end
  end
end

local function place_drawing(box, path, ox, oy)
  local W, H = tex.pagewidth, tex.pageheight
  local image = img.scan({ filename = path })
  image.width, image.height, image.depth = W, H, 0
  local back = node.new(KERN)
  back.kern = -ox
  back.next = img.node(image)
  local row = node.hpack(back)
  local up, down = node.new(KERN), node.new(KERN)
  up.kern, down.kern = -oy, oy - row.height
  -- Typeset glyphs stay in the PDF as invisible text (render mode 3).
  local invisible = node.new("whatsit", "pdf_literal")
  invisible.mode = 1
  invisible.data = "3 Tr"
  up.next, row.next, down.next, invisible.next = row, down, invisible, box.list
  box.list = up
end

local function on_shipout(box)
  M.page = M.page + 1
  local ox, oy = origin()
  scene = new_scene()
  if box.id == VLIST then
    walk_v(box.list, box, ox, oy, nil)
  else
    scene.line = 1
    walk_h(box.list, box, ox, oy + box.height, 1)
  end
  local page = {
    page = M.page, width = pt(tex.pagewidth), height = pt(tex.pageheight),
    glyphs = scene.glyphs, rules = scene.rules, lines = scene.lines,
  }
  if #scene.options > 0 then
    page.options = table.concat(scene.options, ", ")
  end
  local notes = {}
  for _, note in pairs(scene.notes) do
    if note.x and note.from then
      notes[#notes + 1] = note
    end
  end
  table.sort(notes, function(a, b) return a.id < b.id end)
  if #notes > 0 then
    page.notes = notes
  end
  if #scene.right > 0 then
    page.right_notes = scene.right
  end
  if #scene.draw > 0 then
    page.draw = scene.draw
  end
  local erased, any = {}, false
  for _, g in ipairs(scene.glyphs) do
    if g.erased and M.erased_texts[g.erased] then
      erased[tostring(g.erased)] = M.erased_texts[g.erased]
      any = true
    end
  end
  if any then
    page.erased = erased
  end
  local body = json(page)
  local hash = md5.sumhexa(body)
  local base = M.dir .. "/page-" .. M.page
  if read_file(base .. ".json") ~= body then
    write_file(base .. ".json", body)
  end
  local stamp = read_file(base .. ".stamp")
  local drawing = base .. ".jpg"
  if stamp and stamp:gsub("%s+", "") == hash and lfs.isfile(drawing) then
    hide_ink(box.list)
    place_drawing(box, drawing, ox, oy)
  else
    M.waiting = M.waiting + 1
  end
  return box
end

-- Setup and end of run ------------------------------------------------------

-- Distance between the paper's printed lines; the text's line spacing follows it.
local RULE_SPACING = { notebook = "7.1mm", graph = "7.1mm", blank = "7.1mm", wide = "8.7mm", legal = "8.7mm" }

function M.setup(opts)
  M.options = opts
  -- MiKTeX keeps the quotes of a file name with spaces in \jobname: "my notes".
  opts.jobname = opts.jobname:gsub('^"(.*)"$', "%1")
  -- Set by \reroll: which variant of its words to write. Unset on everything else.
  M.variant = luatexbase.attributes["cacography@variant"]
  -- Set by \crossout: 1 the writer's habit, 2 one line, 3 two lines, 4 a scribble.
  M.crossout = luatexbase.attributes["cacography@crossout"]
  -- Set by \erased: which rubbed-out word lies under these glyphs (M.erased_texts).
  M.erased_attr = luatexbase.attributes["cacography@erased"]
  opts.rulespacing = RULE_SPACING[opts.paper] or RULE_SPACING.notebook
  token.set_macro("cacography@rulespacing", opts.rulespacing)
  M.dir = opts.jobname .. ".cacography"
  if not lfs.isdir(M.dir) then
    lfs.mkdir(M.dir)
  end
  luatexbase.add_to_callback("pre_shipout_filter", on_shipout, "cacography.page")
end

-- \cacographypage: a marker in the text, found on whichever page it is shipped with.
function M.page_options(text)
  local n = node.new(WHATSIT, USER)
  n.user_id = PAGE_OPTIONS
  n.type = 115 -- a string
  n.value = text
  node.write(n)
end

-- pgf's \pgfsys@invoke writes a drawing command as a PDF literal, whose text Lua cannot
-- read back; cacography.sty has it call this as well, to leave a readable copy beside it.
function M.literal(text)
  local n = node.new(WHATSIT, USER)
  n.user_id = DRAWN
  n.type = 115
  n.value = text
  node.write(n)
end

-- \erased{old}{new}: what was written first under the glyphs carrying attribute `id`.
M.erased_texts = {}
function M.erased(id, text)
  M.erased_texts[id] = text
end

-- \cacographynote*: kind 1 marks the place in the text, kind 2 the start of the note.
function M.note_mark(id, kind)
  local n = node.new(WHATSIT, USER)
  n.user_id = kind == 1 and NOTE_ANCHOR or kind == 3 and NOTE_RIGHT or NOTE_START
  n.type = 100 -- a number
  n.value = id
  node.write(n)
end

function M.finish()
  local doc = {}
  for k, v in pairs(M.options) do
    doc[k] = v
  end
  doc.pages = M.page + 1 -- the page being finished is shipped after this call
  write_file(M.dir .. "/document.json", json(doc))
end

-- How to start the renderer: CACOGRAPHY_COMMAND, else what `cacography setup` wrote, else
-- `cacography` on the PATH.
local function render_command()
  local cmd = os.getenv("CACOGRAPHY_COMMAND")
  if cmd and cmd ~= "" then
    return cmd
  end
  local cfg = kpse.find_file("cacography-config.lua", "lua") or kpse.find_file("cacography-config.lua", "tex")
  if cfg then
    local ok, t = pcall(dofile, cfg)
    if ok and type(t) == "table" and t.command then
      return t.command
    end
  end
  return "cacography"
end

-- With --shell-escape, draw the pages now, the way bibtex or biber would be run; the next
-- LuaLaTeX run puts them in. Returns whether the renderer finished.
local function render_now()
  local job = M.options.jobname
  if job:find(" ") then
    job = '"' .. job .. '"'
  end
  local line = render_command() .. " render " .. job
  texio.write_nl("term and log", "Package cacography Info: drawing the pages: " .. line)
  texio.write_nl("term", "")
  io.stdout:flush()
  os.setenv("PYTHONUNBUFFERED", "1")
  os.setenv("PYTHONIOENCODING", "utf-8")
  -- Quotes only where a path has spaces: cmd.exe mangles command lines with more than two.
  local a, b, c = os.execute(line)
  local code = (type(a) == "number" and a) or (a == true and 0) or (type(c) == "number" and c) or 1
  return code == 0
end

-- A file every run reads (so latexmk and friends notice when it changes): rewritten when
-- new drawings are ready, which tells them another run is needed.
function M.status_file()
  local path = M.dir .. "/status.tex"
  if not lfs.isfile(path) then
    write_file(path, "% cacography: no pages drawn yet\n")
  end
  return path
end

luatexbase.add_to_callback("stop_run", function()
  if M.dir then
    local doc = read_file(M.dir .. "/document.json")
    if doc then
      write_file(M.dir .. "/document.json", doc:gsub('"pages":%d+', '"pages":' .. M.page))
    end
    local auto = M.options.render ~= "manual"
    if M.waiting > 0 and auto and status.shell_escape == 1 then
      if render_now() then
        write_file(M.dir .. "/status.tex", "% cacography: pages drawn " .. os.date("!%Y-%m-%dT%H:%M:%S") .. "\n")
        texio.write_nl("term and log", string.format(
          "Package cacography Warning: %d of %d pages were just drawn. Rerun to get them into the PDF.",
          M.waiting, M.page))
      else
        texio.write_nl("term and log", string.format(
          "Package cacography Warning: the renderer stopped; %d of %d pages are not drawn. Run `cacography doctor` to see why.",
          M.waiting, M.page))
      end
    elseif M.waiting > 0 then
      local how = auto and "Compile with --shell-escape to draw them automatically, or run" or "Run"
      texio.write_nl("term and log", string.format(
        "Package cacography Warning: %d of %d pages are not drawn yet. %s `cacography render %s`, then LuaLaTeX again.",
        M.waiting, M.page, how, M.options.jobname))
    else
      texio.write_nl("term and log", string.format("Package cacography Info: all %d pages are handwritten.", M.page))
    end
  end
end, "cacography.report")

return M
