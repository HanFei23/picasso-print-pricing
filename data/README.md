# Data

## `booth_inventory.csv`: one row per work at the booth

| column | description |
|---|---|
| `booth_no` | position in the booth price list |
| `bloch` | Bloch catalogue number (`Ba` = Baer number when there is no Bloch number) |
| `title`, `year`, `date_place` | as printed on the price list |
| `medium`, `medium_group` | full medium description and its group (Intaglio, Aquatint, Linocut, Lithograph, Drawing) |
| `image_in`, `sheet_in`, `framed_in` | dimensions in inches |
| `signature`, `series` | e.g. "Signed in pencil, lower right", "Suite Vollard", "Series 156" |
| `ask_usd` | dealer asking price |

## `auction_comps.csv`: one row per auction lot (not committed, see below)

| column | description |
|---|---|
| `bloch` | booth work the lot was matched to |
| `lot_title`, `medium`, `size_cm`, `edition` | as recorded by Artnet |
| `sale_date`, `auction_house`, `sale_lot`, `estimate` | sale details |
| `status` | `Sold`, `Bought In`, `Withdrawn` or `Upcoming` |
| `price_local`, `currency` | hammer price including buyer's premium |
| `price_usd` | converted at the sale-date exchange rate (by Artnet) |
| `exclude` | 1 = suspected data error or reproduction; dropped from the models |
| `note` | free-text flags (e.g. "size likely includes frame") |

Source: Artnet Price Database, queried 2026-10-08 through an institutional subscription. Artnet's terms restrict redistribution, so this file is listed in `.gitignore`.
