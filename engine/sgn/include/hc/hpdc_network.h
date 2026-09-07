/**
 * @file hpdc_network.h
 * @brief HPDC network & distributed ABI - UART, COBS, Lamport clock, watchdog
 * @version 2.0.0
 *
 * Lamport and time types are defined in hc16.h.
 *
 * Depends: hpdc_core.h, hc16.h
 */

#ifndef HPDC_NETWORK_H
#define HPDC_NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

#include "hc/hpdc_core.h"
#include "hc/hc16.h"
#include <stdint.h>
#include <stdbool.h>

/* ============================================================================
 * UART frame
 * ============================================================================ */

#pragma pack(1)

typedef struct {
    uint8_t  sync[2];
    uint8_t  type;
    uint8_t  len;
    uint8_t* payload;
    uint8_t  crc8;
} uart_frame_t;

#pragma pack()

void uart_pack(uint8_t type, const uint8_t* hc_data, uint8_t len, uint8_t* out_frame, uint8_t out_frame_len);
bool uart_parse(const uint8_t* frame, uint8_t frame_len, uart_frame_t* out);

/* ============================================================================
 * COBS
 * ============================================================================ */

uint8_t cobs_encode(const uint8_t* in, uint8_t len, uint8_t* out, uint8_t max_out);
uint8_t cobs_decode(const uint8_t* in, uint8_t len, uint8_t* out, uint8_t max_out);

/* ============================================================================
 * Lamport clock (types in hc16.h)
 * ============================================================================ */

void lamport_update(hc16_lamport_t* local, const hc16_lamport_t* remote);
int lamport_compare(const hc16_lamport_t* a, const hc16_lamport_t* b);

/* ============================================================================
 * Watchdog timer (hc16_time_t in hc16.h)
 * ============================================================================ */

#pragma pack(1)

typedef struct {
    hc16_time_t deadline;
    uint8_t         task_id;
    uint8_t         active;
    void            (*on_timeout)(uint8_t task_id);
} wdt_entry_t;

#pragma pack()

#define SGN_WDT_SLOTS 8

bool is_timeout(const hc16_time_t* deadline, const hc16_time_t* now);
void wdt_start(wdt_entry_t* slots, uint8_t task_id,
                   const hc16_time_t* deadline, void (*callback)(uint8_t));
void wdt_stop(wdt_entry_t* slots, uint8_t task_id);
void wdt_poll(wdt_entry_t* slots, uint8_t n_slots, const hc16_time_t* now);

#ifdef __cplusplus
}
#endif

#endif /* HPDC_NETWORK_H */
