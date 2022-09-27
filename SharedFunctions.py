import math
import datetime
import requests
import json
import hashlib
import random


def mutate_constants(base_pyth_constant, base_uh_oh_multiplier, base_home_advantage_multiplier,
                     base_freshness_coefficient, base_position_weights,
                   base_injury_type_weights, base_spread_coefficient):
    mutated = {
        "pyth_constant": base_pyth_constant,
        "uh_oh_multiplier": base_uh_oh_multiplier,
        "home_advantage_multiplier": base_home_advantage_multiplier,
        "freshness_coefficient": base_freshness_coefficient,
        "position_weights": base_position_weights.copy(),
        "injury_type_weights": base_injury_type_weights.copy(),
        "spread_coefficient": base_spread_coefficient
    }

    # we are just gonna nudge each of these around by 0.0 - 0.1 up or down for each one. This is ~10% randomness in each one (which is a fair amount of genetic drift)
    mutated['pyth_constant'] = base_pyth_constant + ((random.random() - 0.5)/2.5)
    mutated['uh_oh_multiplier'] = base_uh_oh_multiplier + ((random.random() - 0.5)/2)
    mutated['freshness_coefficient'] = base_freshness_coefficient + ((random.random() - 0.5)/2)
    mutated['home_advantage_multiplier'] = base_home_advantage_multiplier + ((random.random() - 0.5)/2)
    mutated['spread_coefficient'] = base_spread_coefficient + ((random.random() - 0.5)*2)

    # for position in base_position_weights.keys():
    #     mutated['position_weights'][position] = base_position_weights[position] + ((random.random() - 0.5)/5)
    #     if mutated['position_weights'][position] < 0:
    #         mutated['position_weights'][position] = 0
    #     elif mutated['position_weights'][position] > 5:
    #         mutated['position_weights'][position] = 5
        # chaos = random.random()
        # if chaos < 0.05:
        #     if base_position_weights[position] > 1:
        #         mutated['position_weights'][position] = base_position_weights[position] - 1
        # elif chaos > 0.95:
        #     if base_position_weights[position] < 5:
        #         mutated['position_weights'][position] = base_position_weights[position] + 1

    return mutated


def evaluate_picks(current_season, week, generation):
    espn_api_base_url = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"
    matchups = get_or_fetch_from_cache(espn_api_base_url + "seasons/" + str(current_season) + "/types/2/weeks/" + str(week) + "/events?lang=en&region=us")
    evaluations = {}

    for prediction_set in generation:
        prediction_set['accuracy_score'] = 0
        prediction_set['spread_score'] = 0
        prediction_set['total_money_won'] = 0
        predictions = prediction_set['predictions']

        for event_link in matchups['items']:
            actual_winner_id = "0"
            event = get_or_fetch_from_cache(event_link['$ref'])
            competition = event['competitions'][0]
            predicted_result = predictions[competition["id"]]
            predicted_winner_id = predicted_result['winner_id']
            competitors = competition['competitors']
            tempspread = 0
            for competitor in competitors:
                score = get_or_fetch_from_cache(competitor['score']['$ref'])
                if tempspread == 0:
                    tempspread = score['value']
                else:
                    tempspread = tempspread - score['value']
                if competitor["winner"]:
                    actual_winner_id = competitor['id']
            # print("predicted/actual: "+ str(predicted_winner_id) + "/" + str(actual_winner_id))

            if(predicted_winner_id == actual_winner_id):
                prediction_set['accuracy_score'] = prediction_set['accuracy_score'] + 1
                predicted_result['chicken_dinner'] = 1
            predicted_result['actual_spread'] = 0 - tempspread
            predicted_result['spread_diff'] = predicted_result['predicted_spread'] - predicted_result['actual_spread']
            prediction_set['spread_score'] = prediction_set['spread_score'] + abs(predicted_result['spread_diff'])
            # if home team won
            if predicted_result['predicted_spread'] > 0:
                # if I predicted they'd win by at least that much, then I would have won a bet on this game
                if predicted_result['vegas_spread'] < predicted_result['actual_spread']:
                    prediction_set['total_money_won'] = prediction_set['total_money_won'] + 100
                    predicted_result['money'] = 'YES'
                # I bet, but they didn't cover
                else:
                    # gotta subtract the loss plus the vig
                    prediction_set['total_money_won'] = prediction_set['total_money_won'] - 110
                    predicted_result['money'] = 'NO'
            # away team won
            else:
                # if I predicted they'd win by at least that much, then I would have won a bet on this game
                if predicted_result['vegas_spread'] > predicted_result['actual_spread']:
                    prediction_set['total_money_won'] = prediction_set['total_money_won'] + 100
                    predicted_result['money'] = 'YES'
                # I bet, but they didn't cover
                else:
                    # gotta subtract the loss plus the vig
                    prediction_set['total_money_won'] = prediction_set['total_money_won'] - 110
                    predicted_result['money'] = 'NO'
        if prediction_set['accuracy_score'] in evaluations:
            evaluations[prediction_set['accuracy_score']].append(prediction_set)
        else:
            evaluations[prediction_set['accuracy_score']] = [prediction_set]

    return evaluations


def get_or_fetch_from_cache(url, directory = "caches"):
    file_key = hashlib.md5(url.encode('UTF-8')).hexdigest()
    try:
        f = open(directory+"/"+file_key, "r")
        data = json.load(f)
        f.close()
        return data
    except:
        # couldn't find that file yet, lets fetch the thing and put it there in the file
        response = requests.get(url)
        data = response.json()
        f = open(directory+"/"+file_key, "x")
        f.write(json.dumps(data, indent=4))
        f.close()
        return data


def rank_impact_weight(total_in_position, rank_in_position):
    # the goal here is to give more weight to positions that have a lot of people in them
    # the 1 rank being out will always be the highest weighted (1.0)
    # the 2 rank being out should depend on how many other people play that position and where they fall in the
    # depth chart
    # the 2 ranked qb being out probably isn't as impactful as the 2 ranked WR
    # my reasoning is the more depth they have in any position means they need to have more of those people on the
    # field at a given time
    # so if you are missing your 2 or 3 wide, it'll have a higher impact than if you are missing your pos 2 kicker
    folks_in_front = rank_in_position - 1
    inv_rank_importance = float(folks_in_front)/float(total_in_position)
    return (1 - inv_rank_importance)/float(total_in_position)


# freshness is represented in terms of 7 days being a "base" freshness of 0, resulting in no adjustment of their overall score
# teams that have a bye week resulting in 14 days since they last played will have a freshness of 1
# while teams that have 4 or 5 days since the last time they played will have a negative freshness
def evaluate_freshness(week, team):
    previous_game_date = None
    this_game_date = None
    team_events = get_or_fetch_from_cache(team['events']["$ref"])
    # lets go get the game they played before this weeks game
    for event in team_events['items']:
        event_info = get_or_fetch_from_cache(event['$ref'])
        season_type = get_or_fetch_from_cache(event_info['seasonType']['$ref'])
        # looking for regular season games
        if season_type['type'] == 2:
            event_week = get_or_fetch_from_cache(event_info['week']['$ref'])
            if int(event_week['number']) == int(week):
                this_game_date = event_info['date']
                break
            else:
                # this assumes that the events listing is in chronological order
                previous_game_date = event_info['date']
    # "date": "2022-08-14T01:00Z",
    if not previous_game_date or not this_game_date:
        return 1
    else:
        previous_datetime = datetime.datetime.strptime(previous_game_date, "%Y-%m-%dT%H:%MZ")
        this_datetime = datetime.datetime.strptime(this_game_date, "%Y-%m-%dT%H:%MZ")
        delta = this_datetime - previous_datetime
        return (delta.days - 7)/7


def generate_picks(current_season, week, pyth_constant, uh_oh_multiplier, home_advantage_multiplier, freshness_coefficient, position_weights, injury_type_weights, spread_coefficient):
    espn_api_base_url = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/"

    depth_charts_file = open("depth_charts.json", "r")
    depth_charts = json.load(depth_charts_file)

    teams = []
    # first lets go get all the teams records from last year so we can start off with something to calculate their pyth with
    team_links = get_or_fetch_from_cache(espn_api_base_url+"teams?limit=32")
    for link in team_links['items']:
        team_info = get_or_fetch_from_cache(link['$ref'])
        # now lets go get their record for last season
        last_season_record = get_or_fetch_from_cache(espn_api_base_url+"seasons/"+str(current_season-1)+"/types/2/teams/"+team_info['id']+"/records/0/?lang=en&region=us")

        # now their record for this season
        this_season_record = get_or_fetch_from_cache(espn_api_base_url+"seasons/"+str(current_season)+"/types/2/teams/"+team_info['id']+"/records/0/?lang=en&region=us")

        # now injuries so we can calculate the uh oh factor
        injuries = get_or_fetch_from_cache(espn_api_base_url + "teams/" + team_info['id'] + "/injuries", "caches/week"+str(week))

        # lets evaluate the teams freshness
        freshness_rating = evaluate_freshness(week, team_info)

        # lets place a limit on this so we don't make a million api calls
        injurycounter = 1
        total_uh_oh_factor = 0
        for item in injuries['items']:
            if(injurycounter < 10):
                # statuses: active, questionable, out
                injury_report = get_or_fetch_from_cache(item['$ref'])
                # get info about the athlete so we know what position he plays/how bad it is
                injured_athlete = get_or_fetch_from_cache(injury_report['athlete']['$ref'])
                rank_impact = 0
                injury_status = injury_report['status']
                # now we crawl over the depth charts to find the team, this player, their rank in their position, etc
                for depth_team in depth_charts:
                    if team_info['id'] == depth_team['id']:
                        for formation in depth_team['depth']['items']:
                            if injured_athlete['position']['abbreviation'].lower() in formation['positions']:
                                position = formation['positions'][injured_athlete['position']['abbreviation'].lower()]
                                position_athletes = position['athletes']
                                for player in position_athletes:
                                    if player['athlete']['$ref'] == injured_athlete['$ref']:
                                        rank_in_position = player['rank']
                                        total_in_position = len(position_athletes)
                                        rank_impact = rank_impact_weight(total_in_position, rank_in_position)
                # print(position_weights[injured_athlete['position']['abbreviation']], rank_impact, injury_status)
                impact = position_weights[injured_athlete['position']['abbreviation']] * rank_impact * injury_type_weights[injury_status]
                total_uh_oh_factor = total_uh_oh_factor + impact
            injurycounter = injurycounter + 1

        this_team = {
            'id': team_info['id'],
            'name': team_info['displayName'],
            'W': this_season_record['stats'][1]['value'],
            'L': this_season_record['stats'][2]['value'],
            'PF': this_season_record['stats'][9]['value'],
            'PA': this_season_record['stats'][10]['value'],
            'GP': this_season_record['stats'][8]['value'],
            'LSW': last_season_record['stats'][1]['value'],
            'LSL': last_season_record['stats'][2]['value'],
            'LSPF': last_season_record['stats'][9]['value'],
            'LSPA': last_season_record['stats'][10]['value'],
            'LSGP': last_season_record['stats'][8]['value'],
            'UHOH': total_uh_oh_factor,
            'FRESHNESS': freshness_rating
        }

        LSWEIGHT = (this_team['LSGP'] - this_team['GP'])/this_team['LSGP']
        if LSWEIGHT < 0:
            LSWEIGHT = 0

        # now using the numbers from above, lets calculate their
        pyth = ((17)*((this_team['PF']+(this_team['LSPF']*LSWEIGHT))**pyth_constant))/((this_team['PF']+(this_team['LSPF']*LSWEIGHT))**pyth_constant + (this_team['PA']+(this_team['LSPA']*LSWEIGHT))**pyth_constant)
        this_team['PYTH'] = pyth
        this_team['SCORE'] = pyth + (freshness_coefficient*freshness_rating) - (total_uh_oh_factor * uh_oh_multiplier)
        teams.append(this_team)

    predictions = {}
    matchups = get_or_fetch_from_cache(espn_api_base_url + "seasons/2022/types/2/weeks/" + week + "/events?lang=en&region=us")
    for event_link in matchups['items']:
        prediction = {
            "winner": "",
            "winner_id": "",
            "teams": []
        }
        event = get_or_fetch_from_cache(event_link['$ref'])
        competition = event['competitions'][0]
        competitors = competition['competitors']
        odds = get_or_fetch_from_cache(competition['odds']['$ref'])
        prediction['name'] = event['name']
        prediction['date'] = event['date']
        for competitor in competitors:
            for team in teams:
                if competitor['id'] == team['id']:
                    this_team = {
                        'id': team['id'],
                        'name': team['name'],
                        'PYTH': team['PYTH'],
                        'UHOH': team['UHOH'],
                        'INJADJ': team['SCORE'],
                        'SCORE': team['SCORE'],
                        'FRESHNESS': team['FRESHNESS']
                    }
                    if(competitor['homeAway'] == 'home'):
                        this_team['SCORE'] = this_team['SCORE'] * home_advantage_multiplier
                    prediction['teams'].append(this_team)

        if prediction['teams'][0]['SCORE'] > prediction['teams'][1]['SCORE']:
            prediction['winner'] = prediction['teams'][0]['name']
            prediction['winner_id'] = prediction['teams'][0]['id']
            prediction['predicted_favorite'] = prediction['teams'][0]['name'] + " -" + str(line_set((prediction['teams'][0]['SCORE'] - prediction['teams'][1]['SCORE'])*spread_coefficient))
        else:
            prediction['winner'] = prediction['teams'][1]['name']
            prediction['winner_id'] = prediction['teams'][1]['id']
            prediction['predicted_favorite'] = prediction['teams'][1]['name'] + " -" + str(line_set((prediction['teams'][1]['SCORE'] - prediction['teams'][0]['SCORE'])*spread_coefficient))

        prediction['vegas_spread'] = odds['items'][0]['spread']
        prediction['predicted_spread'] = line_set(0 - (prediction['teams'][0]['SCORE'] - prediction['teams'][1]['SCORE'])*spread_coefficient)
        if prediction['vegas_spread'] >= prediction['predicted_spread']:
            prediction['bet'] = "away"
        else:
            prediction['bet'] = "home"

        predictions[competition["id"]] = prediction

    # for prediction in predictions:
    #     print(prediction['name'], ",", "Winner:", prediction['winner'], ",", prediction['teams'][0]['name'], ",", round(prediction['teams'][0]['SCORE'], 2), ",", prediction['teams'][1]['name'], ",", round(prediction['teams'][1]['SCORE'],2))
    # print("Constants: pyth, uh_oh, home_advantage: " + str(pyth_constant) + ", " + str(uh_oh_multiplier) + ", " + str(home_advantage_multiplier))
    # f = open("predictions/week" + week + "/" + "test.json", "a")
    return {
        "pyth_constant": pyth_constant,
        "uh_oh_multiplier": uh_oh_multiplier,
        "home_advantage_multiplier": home_advantage_multiplier,
        "position_weights": position_weights,
        "injury_type_weights": injury_type_weights,
        "freshness_coefficient": freshness_coefficient,
        "spread_coefficient": spread_coefficient,
        "predictions": predictions
    }

def line_set(num):
    return round(num * 2) / 2